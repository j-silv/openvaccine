import torch 
from torch.nn import CrossEntropyLoss
from pathlib import Path
from .common import (
    train
)
def save_checkpoint(epoch,
                    model,
                    optimizer,
                    train_losses,
                    val_losses,
                    loss_at_epoch,
                    checkpoint_dir):

    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(exist_ok=True, parents=True)   

        
    checkpoint = dict(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        train_losses=train_losses,
        val_losses=val_losses,
        loss_at_epoch=loss_at_epoch
    )
    torch.save(checkpoint, str(checkpoint_dir / f"{epoch}.pth"))

def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load checkpoint for pretraining step"""

    print("Resuming checkpoint", checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    loss_at_epoch = checkpoint.get("loss_at_epoch", [])

    return epoch, train_losses, val_losses, loss_at_epoch


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

def pretrain(**train_args):
    """Wrapper around training loop for pretraining stage"""

    train(loss_fn=CrossEntropyLoss(),
          calc_loss_fn=calc_loss,
          load_checkpoint_fn=load_checkpoint,
          save_checkpoint_fn=save_checkpoint,
          **train_args)



            