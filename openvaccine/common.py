import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss

def early_stop(losses, patience_losses, current_loss, patience):
    """My implementation for ML training early stopping with patience
    
    losses: a list of saved losses that monotonically decrease
    patience_losses: accumulated losses which are all >= losses[-1]
    current_loss: current loss which will determine if we stop early or not based on other params
    patience: how many losses do we accumulate that are >= losses[-1] before we early stop

    depending on whether or not we have violated patience, the losses 
    and patience_losses lists in place. for example, if the loss is lower than losses[-1]
    then we append to losses and clear the patience_losses list.
    
    returns a tuple of booleans

    stop_training: patience violated, so we should stop training. if false, we might still be in patience interval.
    save_loss: we can save the current loss and patience is reset. used by caller to save checkpoint.
    """

    # first loss, we always save
    if len(losses) == 0:
        return False, True
    
    # case where we either reset patience or have violated patience
    if len(patience_losses) == patience:
        if current_loss >= losses[-1]:
            # violated patience
            return True, False
        else:
            # reset patience
            patience_losses.clear()
            losses.append(current_loss)
            return False, True
    
    # case where we either reset patience or we append to patience list
    elif len(patience_losses) < patience:
        if current_loss >= losses[-1]:
            # not yet violated so append to growing patience list
            patience_losses.append(current_loss)
            return False, False
        else:
            # reset patience
            patience_losses.clear()
            losses.append(current_loss)
            return False, True

    else:
        raise ValueError("size of patience_losses should never be greater than patience")


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

            losses[name] /= num_batches

    model.train()

    return losses["train"], losses["val"]

def mcrmse(preds, labels):
    """Mean columnwise root mean squared error
    
    preds: (B, seq_length, num_regression_targets)
    labels: (B, seq_length, num_regression_targets)
    """
    preds = torch.flatten(preds, 0, 1) # (n, num_regression_targets)
    labels = torch.flatten(labels, 0, 1) # (n, num_regression_targets)

    rmse_per_column = torch.sqrt(torch.mean((preds - labels)**2, dim=0)) # (3, )
    return torch.mean(rmse_per_column)

def train(*,
          model,
          output_dir,
          train_dataloader,
          val_dataloader,
          epochs,
          lr,
          val_interval_per_epoch,
          disable_checkpoint,
          checkpoint,
          early_stopping,
          patience,
          loss_fn,
          calc_loss_fn,
          load_checkpoint_fn,
          save_checkpoint_fn,
          validation_batch_size,
          ):
    """Common training loop for both pretraining and finetuning"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not disable_checkpoint:
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    if checkpoint:
        starting_epoch, train_losses, val_losses, loss_at_step = \
            load_checkpoint_fn(model, optimizer, checkpoint)
    else:
        starting_epoch = 0
        train_losses = []
        val_losses = []
        loss_at_step = []

    # either way we start at fresh patience cause if the checkpoint is saved
    # the patience is reset
    patience_losses = []

    for epoch in range(starting_epoch, epochs):

        if val_interval_per_epoch and epoch % val_interval_per_epoch == 0:
            train_loss, val_loss = get_dataloader_losses(model,
                                                        loss_fn,
                                                        calc_loss_fn,
                                                        train_dataloader,
                                                        val_dataloader,
                                                        device,
                                                        num_batches=validation_batch_size)
            
            save_loss = True
            if early_stopping:
                stop_training, save_loss = early_stop(val_losses, patience_losses, val_loss, patience)
                if stop_training:
                    print(f"EARLY STOPPING WITH VAL LOSS {val_loss:.4f} - best val loss is {val_losses[-1]:.4f}")
                    return
            
            print(f"Epoch ({epoch}/{epochs}) | Avg Train Loss {train_loss:.4f} |  Avg Val Loss {val_loss:.4f}", end="")

            if save_loss:
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                loss_at_step.append(epoch)
                plot_loss_curves(train_losses, val_losses, loss_at_step, output_dir)

                if not disable_checkpoint:
                    save_checkpoint_fn( epoch,
                                        model,
                                        optimizer,
                                        train_losses,
                                        val_losses,
                                        loss_at_step,
                                        checkpoint_dir)
                    print(" (checkpoint saved)")
                else:
                    print("")

            else:
                print(f" patience ({len(patience_losses)}/{patience})")



        model.train()
        avg_batch_loss = 0

        for batch, (sequence, y1, y2, y3) in enumerate(train_dataloader):

            sequence = sequence.to(device)
            labels = torch.stack((y1, y2, y3), dim=-1).to(device)
            loss = calc_loss_fn(model, sequence, loss_fn, labels)
        
            avg_batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_batch_loss /= len(train_dataloader)

        print(f"Epoch ({epoch}/{epochs}) | Avg Train Batch Loss {avg_batch_loss:.4f}")