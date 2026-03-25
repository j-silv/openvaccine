import torch
import torch.nn as nn

class RNAModel(nn.Module):
    def __init__(self, embd=4, vocab_len=4, seq_len=107, out_dim=3):
        super().__init__()
        self.embd = embd
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab_len, embd)
        self.pos_emb = nn.Embedding(seq_len, embd)

        # Simple NN to test training 
        self.layers = nn.Sequential(
            nn.Linear(embd, embd),
            nn.ReLU(),
            nn.Linear(embd, embd),
            nn.ReLU(),                                            
            nn.Linear(embd, out_dim) # predicts all regression targets for each letter
        )

    def forward(self, x):
        pos = self.pos_emb(torch.arange(self.seq_len)) 
        emb = self.tok_emb(x)

        result = pos + emb

        return self.layers(result)
        



