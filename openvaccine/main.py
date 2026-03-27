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

    json_data = load_data(max_lines=10)

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
        drop_rate=0.1, # dropout probaility for regularization
        qkv_bias=False, # use bias in query, key, value linear projections
        mask_token=tokenizer.encode("m") # for masking in forward pass
    )

    model = RNAModel(cfg)

    pretrain(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()