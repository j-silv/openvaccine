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
from pathlib import Path



def get_args(bert_model_choices):
    parser = argparse.ArgumentParser(
        description="OpenVaccine RNA stability prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )  
    
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint to load. If None, don't load checkpoint", default=None)
    parser.add_argument("-n", "--num_samples", type=int, help="Number of total data samples to load. If None, load all samples", default=None)
    parser.add_argument("-m", "--model", default="base", choices=bert_model_choices, help="Size of BERT model")
    parser.add_argument("-b", "--batch_size", default=8, type=int, help="Model training batch size")

    parser.add_argument("--data_split", type=float, default=0.9, help="Train/validation data split ratio")
    parser.add_argument("--seq_len", type=int, default=107, help="Max size of RNA sequence")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout probability (regularization)")
    parser.add_argument("--qkv_bias", action="store_true", help="Use bias in query, key, and value linear projects")
    parser.add_argument("--mask_percent", type=float, default=0.15, help="What percentage of tokens in MLM to randomly select")
    parser.add_argument("--mask_prob", type=float, default=0.8, help="What probability should selected tokens be randomly set to mask_id")
    parser.add_argument("--random_prob", type=float, default=0.1, help="What probability should selected tokens be randomly set to a random token between [0, vocab_len-2]")
    parser.add_argument("--same_prob", type=float, default=0.1, help="What probability should selected tokens not be changed")

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train on")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--val_interval_per_epoch", type=int, default=0, help="How many epochs to pass before running validation (and checkpointing if enabled)")
    parser.add_argument("--disable_checkpoint", action="store_true", help="If saving checkpoints is enabled")

    parser.add_argument("--validation_batch_size", type=int, default=50, help="Number of batches to get validation performance")

    parser.add_argument("--early_stopping", action="store_true", help="If validation loss > last validation loss, quit training")
    parser.add_argument("--patience", type=int, default=0, help="Give this many validation loss checks before actually early stopping")

    subparsers = parser.add_subparsers(dest="stage", metavar="STAGE", help="Training stage to run",
                                       required=True)
    
    subparser = subparsers.add_parser('pretrain', help="Pretrain BERT-style RNA language model on MLM task")
    subparser = subparsers.add_parser('finetune', help="Finetune BERT-style RNA language model on stability regression task")

    args = parser.parse_args()

    return args

def main(): 

    bert_model_sizes = {
        "base": dict(
            embd=768,
            n_layers=12,
            n_heads=12
        ),
        "large": dict(
            embd=1024,
            n_layers=24,
            n_heads=16
        )
    }

    args = get_args(bert_model_sizes.keys())

    random.seed(123)
    torch.manual_seed(123)
    
    tokenizer = RNATokenizer()

    # inspired from Build an LLM from Scratch
    model_cfg = dict(
        vocab_len=tokenizer.vocab_size, # 4 nucleotides + mask + pad token
        seq_len=args.seq_len, # 107 for training data as per Kaggle competition
        drop_rate=args.drop_rate, # dropout probaility for regularization
        qkv_bias=args.qkv_bias, # use bias in query, key, value linear projections
        mask_token=tokenizer.l2t["m"], # for masking in forward pass
        mask_percent=args.mask_percent, # what percentage of tokens to randomly select
        mask_prob=args.mask_prob, # what probability should they be set to mask_id
        random_prob=args.random_prob, # what probability should they be set to a random token [0, vocab_len-2)
        same_prob=args.same_prob, # what probability should they not change
        num_regression_targets=3 # how many regression targets
    )

    model_cfg.update(bert_model_sizes[args.model])

    model = RNAStabilityClassifier(model_cfg)


    if args.stage == "infer":
        raise NotImplementedError("Infer stage not yet implemented")
        # return so we don't do remaining training steps after this if statement
        return


    json_data = load_data(max_lines=args.num_samples)
    train_dataset, val_dataset = split_data(json_data, data_split=args.data_split)
    train_dataloader = create_dataloader(train_dataset, tokenizer, batch_size=args.batch_size)
    val_dataloader = create_dataloader(val_dataset, tokenizer, batch_size=args.batch_size)

    train_args = dict(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        val_interval_per_epoch=args.val_interval_per_epoch,
        checkpoint=args.checkpoint,
        disable_checkpoint=args.disable_checkpoint,
        early_stopping=args.early_stopping,
        patience=args.patience,
        validation_batch_size=args.validation_batch_size
    )

    date_and_time_id = dt.datetime.now().strftime("%m%d%y-%H%M%S")

    if args.stage == "pretrain":
        train_fn = pretrain
        output_dir = f"outputs/pretrain/{date_and_time_id}"
        train_model = model.bert

    elif args.stage == "finetune":
        train_fn = finetune
        output_dir = f"outputs/finetune/{date_and_time_id}"
        train_model = model
        
    else:
        raise ValueError("Unexpected stage", args.stage)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "command.txt", "w") as f:
        f.write(" ".join(sys.argv))

    train_fn(model=train_model, output_dir=output_dir, **train_args)

if __name__ == "__main__":
    main()