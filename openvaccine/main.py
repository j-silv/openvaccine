from .data import (
    load_data,
    create_dataloader,
    split_data,
    RNATokenizer
)

from .model import RNAModel
from .train import train

def main():
    json_data = load_data(max_lines=10)

    tokenizer = RNATokenizer()
    
    train_dataset, val_dataset = split_data(json_data)

    train_dataloader = create_dataloader(train_dataset, tokenizer)
    val_dataloader = create_dataloader(val_dataset, tokenizer)

    model = RNAModel()

    train(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()