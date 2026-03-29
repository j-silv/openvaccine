import torch 
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss
import matplotlib.pyplot as plt
from pathlib import Path


def get_dataset_losses(model, loss_fn, train_dataloader, val_dataloader, device):
    model.eval()

    losses = dict(train=0.0, val=0.0)

    with torch.no_grad():
        for name, data_loader in zip(losses, [train_dataloader, val_dataloader]):
            for batch, (sequence, y1, y2, y3) in enumerate(data_loader):
                sequence = sequence.to(device)

                masked_tokens_idx, logits, _ = model(sequence) # B, T, 3
                masked_tokens_predicted = logits[masked_tokens_idx]
                masked_tokens_target = sequence[masked_tokens_idx]
                loss = loss_fn(masked_tokens_predicted, masked_tokens_target)

                losses[name] += loss.item()

            losses[name] /= len(data_loader)

    model.train()

    return losses["train"], losses["val"]

def plot_loss_curves(train_losses, val_losses, loss_at_step):
    output_dir = Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    plt.figure()
    plt.plot(loss_at_step, train_losses, label="train")
    plt.plot(loss_at_step, val_losses, label="val")
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title("Loss curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves.png")

def train(model, train_dataloader, val_dataloader,
             epochs=30000,
             lr=0.0001,
             val_interval_per_step=100,
             load_checkpoint=None,
             checkpoint_interval=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    checkpoint_path = Path("outputs/checkpoints")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True, parents=True)

    if load_checkpoint:
        print("Resuming checkpoint", load_checkpoint)
        load_checkpoint = Path(load_checkpoint)
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        loss_at_step = checkpoint.get("loss_at_step", [])
    else:
        starting_epoch = 0
        global_step = 0
        train_losses = []
        val_losses = []
        loss_at_step = []

    for epoch in range(starting_epoch, epochs):
        model.train()

        avg_batch_loss = 0
        for batch, (sequence, y1, y2, y3) in enumerate(train_dataloader):
            
            if global_step % val_interval_per_step == 0:
                train_loss, val_loss = get_dataset_losses(model, loss_fn, train_dataloader, val_dataloader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                loss_at_step.append(global_step)
                plot_loss_curves(train_losses, val_losses, loss_at_step)
                print(f"Epoch ({epoch}/{epochs}) | Step ({global_step}) | Train Loss {train_loss:.4f} |  Val Loss {val_loss:.4f}")


            if global_step % checkpoint_interval == 0:
                print(f"Saving checkpoint at step {global_step}")
                checkpoint = dict(
                    epoch=epoch,
                    global_step=global_step,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    train_losses=train_losses,
                    val_losses=val_losses,
                    loss_at_step=loss_at_step
                )
                torch.save(checkpoint, checkpoint_path / f"checkpoint_{global_step}.pth")

            sequence = sequence.to(device)
        
            masked_tokens_idx, logits, _ = model(sequence) # B, T, 3
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

            global_step += 1

        avg_batch_loss /= len(train_dataloader)

        print(f"Epoch ({epoch}/{epochs}) | Loss {avg_batch_loss:.4f}")



            