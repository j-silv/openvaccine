"""
Transformer architecture implementation heavily taken from:

Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
"""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embd"], 4*cfg["embd"]),
            nn.GELU(),
            nn.Linear(4*cfg["embd"], cfg["embd"])
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,
                 num_heads, qkv_bias=False):
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x) # b, num_tokens, d_out
        queries = self.W_query(x) # b, num_tokens, d_out
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # b, num_heads, num_tokens, head_dim
        queries = queries.transpose(1, 2) # b, num_heads, num_tokens, head_dim
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) # b, num_heads, num_tokens, num_tokens

        # non-casual attention, tokens attend over all other tokens
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # b, num_heads, num_tokens, num_tokens

        # b, num_heads, num_tokens, head_dim -> b, num_tokens, num_heads, head_dim
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["embd"],
            d_out = cfg["embd"],
            num_heads = cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["embd"])
        self.norm2 = nn.LayerNorm(cfg["embd"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


def mask_tokens(tokens, mask_token, vocab_len, change_all_to_mask=True, test_without_masking=False, test_without_sampling=False):
    # TODO I suspect there is a bug here... loss stays at around 1.0-1.3 when trying to overfit. Might just be capacity of model.
    # should try removing the 80%, 10%, and 10% thing and just try replacing the indexed tokens with the mask token
    batch_len, seq_len = tokens.shape

    # to overfit without the random sampling
    if test_without_masking:
        selected_idxs = torch.arange(seq_len, device=tokens.device)
    else:
        # TODO replace with simpler torch.rand < 0.15
        # randomly choose 15% of the tokens
        selected_idxs = torch.multinomial(
            # every index has equal probability of being selected
            torch.ones_like(tokens, dtype=torch.float32, device=tokens.device),
            num_samples=int(0.15*seq_len),
            replacement=False,
        )
    
    # now that we know which indices to select and later we determine which action to perform,
    # let's select the actual values now
    # we broadcast the first index array to selected_idxs size so we can select the appropiate elements
    masked_tokens_idx = (torch.arange(batch_len, device=tokens.device)[:, None], selected_idxs)

    if test_without_sampling:
        masked_tokens = tokens[masked_tokens_idx]
    else:
        # of those 15% tokens selected:
        # - set to mask token with 80% probability
        # - set to random token with 10% probability
        # - don't change with 10% probability
        # TODO replace with simpler torch.rand
        if change_all_to_mask:
            probs = [1.0]
        else:
            probs = [0.8, 0.1, 0.1]
        action = torch.multinomial(
            torch.tensor(probs, device=tokens.device),
            num_samples=selected_idxs.numel(),
            replacement=True
        )
        action = action.reshape(selected_idxs.shape)

        masked_tokens = torch.where(action == 0, mask_token.to(tokens.device), action)

        random_tokens = torch.randint(vocab_len, size=action.shape, device=tokens.device)

        masked_tokens = torch.where(action == 1, random_tokens, masked_tokens)
        masked_tokens = torch.where(action == 2, tokens[masked_tokens_idx], masked_tokens)
    
    return masked_tokens_idx, masked_tokens

class RNAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.vocab_len = cfg["vocab_len"]
        self.mask_token = cfg["mask_token"]

        self.tok_emb = nn.Embedding(cfg["vocab_len"], cfg["embd"])
        self.pos_emb = nn.Embedding(cfg["seq_len"], cfg["embd"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["embd"])

        # -2 because we ignore mask and pad tokens
        self.out_head = nn.Linear(cfg["embd"], cfg["vocab_len"]-2, bias=False)

    def forward(self, tokens):
        batch_len, seq_len = tokens.shape
        vocab_len = self.pos_emb.weight.shape[0]

        if seq_len != vocab_len:
            raise ValueError(f"Unexpected number of elements {seq_len} compared to pos embedding {vocab_len}")

        # vocab_len-1 because the last token is the pad token
        masked_tokens_idx, masked_tokens = mask_tokens(tokens, self.mask_token, self.vocab_len-1)

        # create copy of original tokens since original tokens we use for cross entropy
        masked = tokens.detach().clone()
        masked[masked_tokens_idx] = masked_tokens

        emb = self.tok_emb(masked)
        pos = self.pos_emb(torch.arange(tokens.shape[-1], device=tokens.device)) 

        x = pos + emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return masked_tokens_idx, logits
        



