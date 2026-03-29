from .data import (
    load_data,
    create_dataloader,
    split_data,
    RNATokenizer
)

from .model import RNAModel
from .train import pretrain
import torch
import random

def main(): 
    torch.manual_seed(123)

    json_data = load_data(max_lines=2)

    tokenizer = RNATokenizer()
    
    train_dataset, val_dataset = split_data(json_data)

    train_dataloader = create_dataloader(train_dataset, tokenizer,batch_size=8)
    val_dataloader = create_dataloader(val_dataset, tokenizer, batch_size=8)

    # inspired from Build an LLM from Scratch
    cfg = dict(
        embd=768, # transformer embedding dimension
        vocab_len=tokenizer.vocab_size, # 4 nucleotides + mask + pad token
        seq_len=107, # 107 for training data as per Kaggle competition
        out_dim=3, # 3 regression targets for each letter in sequence
        n_heads=12, # number of heads in multi-head attention
        n_layers=12, # number of transformer blocks
        drop_rate=0.0, # dropout probaility for regularization
        qkv_bias=False, # use bias in query, key, value linear projections
        mask_token=tokenizer.l2t["m"], # for masking in forward pass
        mask_percent=0.15, # what percentage of tokens to randomly select
        mask_prob=0.8, # what probability should they be set to mask_id
        random_prob=0.1, # what probability should they be set to a random token [0, vocab_len-2)
        same_prob=0.1 # what probability should they not change
    )

    BERT_BASE = dict(
        embd=768,
        n_layers=12,
        n_heads=12
    )

    BERT_LARGE = dict(
        embd=1024,
        n_layers=24,
        n_heads=16
    )

    cfg.update(BERT_BASE)

    model = RNAModel(cfg)

    pretrain(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()