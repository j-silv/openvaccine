import torch 
from torch.nn import CrossEntropyLoss
from pathlib import Path
from .common import (
    train
)
def save_checkpoint(global_step,
                    epoch,
                    model,
                    optimizer,
                    train_losses,
                    val_losses,
                    loss_at_step,
                    checkpoint_dir):
    print(f"Saving checkpoint at step {global_step}")


    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(exist_ok=True, parents=True)   

        
    checkpoint = dict(
        epoch=epoch,
        global_step=global_step,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        train_losses=train_losses,
        val_losses=val_losses,
        loss_at_step=loss_at_step
    )
    torch.save(checkpoint, str(checkpoint_dir / f"{global_step}.pth"))

def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load checkpoint for pretraining step"""

    print("Resuming checkpoint", checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    starting_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    loss_at_step = checkpoint.get("loss_at_step", [])

    return starting_epoch, global_step, train_losses, val_losses, loss_at_step


def calc_loss(model, sequence, loss_fn, labels):
    """Output and loss calculation for pretraining step
    
    Note that the regression labels are ignored for pretraining
    """
    masked_tokens_idx, logits, _ = model(sequence) # B, T, 3
    masked_tokens_predicted = logits[masked_tokens_idx]
    masked_tokens_target = sequence[masked_tokens_idx]

    if masked_tokens_predicted.numel() == 0 and masked_tokens_target.numel() == 0.0:
        raise ValueError("Zero tokens are selected which means loss is technically 0.0. "
                        "Change cfg['mask_percent'] to something > zero")

    loss = loss_fn(masked_tokens_predicted, masked_tokens_target)
    
    return loss

def pretrain(model,
             train_dataloader,
             val_dataloader,
             output_dir,
             epochs=30000,
             lr=0.0001,
             val_interval_per_step=5,
             checkpoint_dir=None,
             checkpoint_interval=5):
    """Wrapper around training loop for pretraining stage"""

    train(
        model,
        train_dataloader,
        val_dataloader,
        output_dir,
        CrossEntropyLoss(),
        calc_loss,
        load_checkpoint,
        save_checkpoint,
        checkpoint_dir,
        epochs=epochs,
        lr=lr,
        val_interval_per_step=val_interval_per_step,
        checkpoint_interval=checkpoint_interval)



            