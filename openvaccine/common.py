import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss

def plot_loss_curves(train_losses, val_losses, loss_at_step, output_dir):
    """Train and validation loss curves"""

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(loss_at_step, train_losses, label="train")
    plt.plot(loss_at_step, val_losses, label="val")
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title("Loss curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves.png")
    plt.close()


def get_dataloader_losses(model,
                          loss_fn,
                          calc_loss_fn,
                          train_dataloader,
                          val_dataloader,
                          device,
                          num_batches=None):
    """Get losses over dataloaders for either pretrain/finetune"""

    model.eval()

    losses = dict(train=0.0, val=0.0)

    with torch.no_grad():
        for name, data_loader in zip(losses, [train_dataloader, val_dataloader]):

            if not num_batches:
                num_batches = len(data_loader)

            for batch, (sequence, y1, y2, y3) in enumerate(data_loader):
                sequence = sequence.to(device)
                labels = torch.stack((y1, y2, y3), dim=-1).to(device)

                loss = calc_loss_fn(model, sequence, loss_fn, labels)

                losses[name] += loss.item()

                if batch == num_batches-1:
                    break

            losses[name] /= len(data_loader)

    model.train()

    return losses["train"], losses["val"]


def train(model,
          train_dataloader,
          val_dataloader,
          output_dir,
          loss_fn,
          calc_loss_fn,
          load_checkpoint_fn,
          save_checkpoint_fn,
          checkpoint_dir,
          epochs=30000,
          lr=0.0001,
          val_interval_per_step=100,
          load_checkpoint=None,
          checkpoint_interval=100,
          early_stopping=False):
    """Common training loop for both pretraining and finetuning"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
        starting_epoch, global_step, train_losses, val_losses, loss_at_step = \
            load_checkpoint_fn(model, optimizer, checkpoint_dir)
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

            if val_interval_per_step and global_step % val_interval_per_step == 0:
                train_loss, val_loss = get_dataloader_losses(model,
                                                             loss_fn,
                                                             calc_loss_fn,
                                                             train_dataloader,
                                                             val_dataloader,
                                                             device,
                                                             num_batches=50)
                
                if len(val_losses) > 0 and early_stopping:
                    if val_loss > val_losses[-1]:
                        print(f"EARLY STOPPING WITH VAL LOSS {val_loss:.4f} - best val loss is {val_losses[-1]:.4f}")
                        return

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                loss_at_step.append(global_step)
                plot_loss_curves(train_losses, val_losses, loss_at_step, output_dir)
                print(f"Epoch ({epoch}/{epochs}) | Step ({global_step}) | Train Loss {train_loss:.4f} |  Val Loss {val_loss:.4f}")


            if checkpoint_interval and global_step % checkpoint_interval == 0:
                save_checkpoint_fn(global_step,
                                    epoch,
                                    model,
                                    optimizer,
                                    train_losses,
                                    val_losses,
                                    loss_at_step,
                                    Path(output_dir) / "checkpoints")

            sequence = sequence.to(device)
            labels = torch.stack((y1, y2, y3), dim=-1).to(device)
            loss = calc_loss_fn(model, sequence, loss_fn, labels)
        
            avg_batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        avg_batch_loss /= len(train_dataloader)

        print(f"Epoch ({epoch}/{epochs}) | Loss {avg_batch_loss:.4f}")