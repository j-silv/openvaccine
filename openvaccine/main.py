from .data import (
    load_data,
    create_dataloader,
    split_data,
    RNATokenizer
)

def main():
    json_data = load_data(max_lines=10)

    tokenizer = RNATokenizer()
    
    train_dataset, val_dataset = split_data(json_data)

    train_dataloader = create_dataloader(train_dataset, tokenizer)
    val_dataloader = create_dataloader(val_dataset, tokenizer)


if __name__ == "__main__":
    main()