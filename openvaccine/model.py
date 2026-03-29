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


def mask_tokens(tokens, mask_id, vocab_len, *,
                percent=0.15, mask_prob=0.80,
                random_prob=0.10, same_prob=0.10):
    """Select a percentange of tokens to mask out (follows Task #1 in BERT)
    
    percent: what percentage of tokens to randomly select (per sequence if tokens is batched)
             the implementation does not technically always select this percentage of tokens
             but rather there is a "percent" probability that the ith token is chosen
    mask_prob: of the selected tokens, with what probability should they be set to mask_id
    random_prob: of the selected tokens, with what probability should they be set to a random token [0, vocab_len)
    same_prob: of the selected tokens, with what probability should they not change
    """

    if not torch.allclose(torch.tensor(mask_prob + random_prob + same_prob), torch.tensor(1.0)):
        raise ValueError("mask_prob + random_prob + same_prob needs to sum to 1")
    
    # because we don't want to modify the original tokens which will be used
    # as the labels for cross entropy
    masked_tokens = tokens.detach().clone()

    # select a subset of the tokens
    selected_idxs = torch.rand(tokens.shape) < percent

    original_tokens = masked_tokens[selected_idxs]
    probs = torch.rand(original_tokens.shape)
    
    # set to mask 
    set_to_mask = probs < mask_prob
    original_tokens[set_to_mask] = mask_id
    
    # randomly select a token id
    set_to_random = (probs >= mask_prob) & (probs < mask_prob + random_prob)
    original_tokens[set_to_random] = torch.randint(vocab_len,
                                                   size=(set_to_random.sum().item(), ), # type: ignore
                                                   device=tokens.device) 
    
    # for the rest do nothing, but update cloned tensor
    masked_tokens[selected_idxs] = original_tokens

    return selected_idxs, masked_tokens


class RNAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.vocab_len = cfg["vocab_len"]
        self.mask_token = cfg["mask_token"]
        self.mask_percent=cfg["mask_percent"]
        self.mask_prob=cfg["mask_prob"]
        self.random_prob=cfg["random_prob"]
        self.same_prob=cfg["same_prob"]

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

        # vocab_len-2 because the last two tokens are MASK and PAD tokens
        masked_tokens_idx, masked_tokens = mask_tokens(
            tokens, self.mask_token, self.vocab_len-2,
            percent=self.mask_percent, mask_prob=self.mask_prob,
            random_prob=self.random_prob, same_prob=self.same_prob)

        emb = self.tok_emb(masked_tokens)
        pos = self.pos_emb(torch.arange(tokens.shape[-1], device=tokens.device)) 

        x = pos + emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        # return output before final layer for downstream tasks
        return masked_tokens_idx, logits, x 
        

class RNAStabilityClassifier(nn.Module):
    """Wrapper class around RNAModel so that we can load pretrained weights"""
    def __init__(self, cfg):
        self.bert = RNAModel(cfg)
        self.classifier = nn.Linear(cfg["embd"], cfg["num_regression_targets"])

    def forward(self, x):
        x = self.bert(x)
        x = self.classifier(x)
        return x
