import torch 
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss

def pretrain(model, train_dataloader, val_dataloader, epochs=30000, lr=0.0001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)


    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        batch_losses = []
        avg_batch_loss = 0
        for batch, (sequence, y1, y2, y3) in enumerate(train_dataloader):
            B, T = y1.shape

            sequence = sequence.to(device)
        
            masked_tokens_idx, logits = model(sequence) # B, T, 3

            masked_tokens_predicted = logits[masked_tokens_idx]
            masked_tokens_target = sequence[masked_tokens_idx]

            if masked_tokens_predicted.numel() == 0 and masked_tokens_target.numel() == 0.0:
                raise ValueError("Zero tokens are selected which means loss is technically 0.0. "
                                 "Change cfg['mask_percent'] to something > zero")
            else:
                loss = loss_fn(masked_tokens_predicted, masked_tokens_target)

            avg_batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_batch_loss /= len(train_dataloader)
        batch_losses.append(avg_batch_loss)

        print(f"Epoch ({epoch}/{epochs}) | Loss {avg_batch_loss:.4f}")



            