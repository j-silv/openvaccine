from .data import (
    load_data,
    create_dataloader,
    split_data,
    RNATokenizer
)

from .model import RNAStabilityClassifier
from .pretrain import pretrain
from .finetune import finetune
import torch
import random
import argparse
import sys
import datetime as dt


def get_args():
    parser = argparse.ArgumentParser(
        description="OpenVaccine RNA stability prediction",
        formatter_class=argparse.RawTextHelpFormatter
    )  
    
    parser.add_argument("-c", "--checkpoint_dir", help="Path to checkpoint to load", default=None)
    parser.add_argument("-n", "--num_samples", type=int, help="Number of total data samples to load. If None, load all samples.", default=None)
    subparsers = parser.add_subparsers(dest="stage", metavar="STAGE", help="Training stage to run",
                                       required=True)
    
    subparser = subparsers.add_parser('pretrain', help="Pretrain BERT-style RNA language model on MLM task")
    subparser = subparsers.add_parser('finetune', help="Finetune BERT-style RNA language model on stability regression task")

    args = parser.parse_args()

    return args

def main(): 

    args = get_args()
        
    torch.manual_seed(123)

    json_data = load_data(max_lines=args.num_samples)

    tokenizer = RNATokenizer()
    
    train_dataset, val_dataset = split_data(json_data)

    train_dataloader = create_dataloader(train_dataset, tokenizer,batch_size=32)
    val_dataloader = create_dataloader(val_dataset, tokenizer, batch_size=32)

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
        mask_token=tokenizer.l2t["m"], # for masking in forward pass
        mask_percent=0.15, # what percentage of tokens to randomly select
        mask_prob=0.8, # what probability should they be set to mask_id
        random_prob=0.1, # what probability should they be set to a random token [0, vocab_len-2)
        same_prob=0.1, # what probability should they not change
        num_regression_targets=3 # how many regression targets
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

    model = RNAStabilityClassifier(cfg)

    train_args = dict(
        epochs=50,
        lr=0.0001,
        val_interval_per_step=len(train_dataloader),
        checkpoint_dir=None,
        checkpoint_interval=len(train_dataloader),
        early_stopping=False,
    )

    date_and_time_id = dt.datetime.now().strftime("%m%d%y-%H%M%S")

    if args.checkpoint_dir:
        train_args["checkpoint_dir"] = args.checkpoint_dir

    if args.stage == "pretrain":
        pretrain(model.bert,
                 train_dataloader,
                 val_dataloader,
                 f"outputs/pretrain/{date_and_time_id}",
                 **train_args
                 )
        
    elif args.stage == "finetune":
        finetune(model,
                 train_dataloader,
                 val_dataloader,
                 f"outputs/finetune/{date_and_time_id}",
                 **train_args
                 )
    else:
        raise ValueError("Unexpected stage", args.stage)

if __name__ == "__main__":
    main()