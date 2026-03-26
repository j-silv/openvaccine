import torch 
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss

def pretrain(model, train_dataloader, val_dataloader, epochs=300, lr=0.001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    for epoch in range(epochs):
        model.train()

        batch_losses = []
        avg_batch_loss = 0
        for batch, (sequence, y1, y2, y3) in enumerate(train_dataloader):
            B, T = y1.shape
            masked_tokens_idx, logits = model(sequence) # B, T, 3
            
            # TODO replace with actual pretraining task, not the regression task
            
            # because the whole sequence is processed but we only have targets for the beginning nucleotides
            y1_pred, y2_pred, y3_pred = logits[:, :T, 0], logits[:, :T, 1], logits[:, :T, 2]

            loss = loss_fn(y1_pred, y1) + loss_fn(y2_pred, y2) + loss_fn(y3_pred, y3)

            avg_batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_batch_loss /= len(train_dataloader)
        batch_losses.append(avg_batch_loss)

        print(f"Epoch ({epoch}/{epochs}) | Loss {avg_batch_loss:.4f}")



            