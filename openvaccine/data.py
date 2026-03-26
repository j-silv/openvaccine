import json
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt

class RNATokenizer:
    """Very simple tokenizer which converts RNA strings into PyTorch tensors and vice versa"""
    def __init__(self):

        self.l2t = dict() # letter to token
        self.t2l = dict() # token to letter

        letters = ["A", "C", "G", "U", "m", "p"] # m for mask, p for pad

        for i, letter in enumerate(letters):
            self.l2t[letter] = i
            self.t2l[i] = letter

        self.vocab_size = len(self.l2t)

    def encode(self, string):
        result = []

        for l in string:
            result.append(self.l2t[l])
        
        return torch.tensor(result)
    
    def decode(self, tokens):
        result = []

        for t in tokens:
            result.append(self.t2l[t.item()])


        return "".join(result)


def load_data(path="data/train.json", max_lines=None):
    """Read in each json line fully up to max_lines if specified"""
    json_data = []

    lines_read = 0
    with open(path, "r") as f:
        for line in f:
        
            if max_lines and lines_read == max_lines:
                break

            decoded = json.loads(line)
            json_data.append(decoded)
            lines_read += 1

    return json_data


class OpenVaccineDataset(Dataset):
    def __init__(self, json_data, tokenizer):
        self.sequences = [] # RNA molecule strings
        self.reactivity = []
        self.deg_Mg_pH10 = []
        self.deg_Mg_50C = []

        for json_dict in json_data:
            sequence = json_dict["sequence"]
            self.sequences.append(tokenizer.encode(sequence))
            self.reactivity.append(torch.tensor(json_dict["reactivity"]))
            self.deg_Mg_pH10.append(torch.tensor(json_dict["deg_Mg_pH10"]))
            self.deg_Mg_50C.append(torch.tensor(json_dict["deg_Mg_50C"]))

    def __getitem__(self, idx):
        return self.sequences[idx], self.reactivity[idx], self.deg_Mg_pH10[idx], self.deg_Mg_50C[idx]
    
    def __len__(self):
        return len(self.sequences)

def split_data(json_data, data_split=0.9):
    data_split_index = int(data_split*len(json_data))

    train_data = json_data[:data_split_index]
    val_data = json_data[data_split_index:]

    return train_data, val_data

def create_dataloader(json_data, tokenizer, batch_size=2, shuffle=False):
    """Create the PyTorch dataset and the associated dataloader"""

    dataset = OpenVaccineDataset(json_data, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle) 
    return dataloader


def get_sequence_info(dataset, plot=False):
    """Return and optionally plot the distribution of RNA sequence lengths"""

    seq_lens = []
    for sequence, _ in dataset:
        seq_lens.append(len(sequence))

    if plot:
        plt.hist(seq_lens)
        plt.title("Distribution of RNA sequence lengths")
        plt.xlabel("Length")
        plt.ylabel("Count")
        plt.show()


    return seq_lens

