import torch 
from pathlib import Path
from torch.nn import MSELoss
from .common import (
    train
)

def calc_loss(model, sequence, loss_fn, labels):
    """Output and loss calculation for finetuning step
    
    Note regression labels is required finetuning but not for pretraining
    """
    stability_predicted = model(sequence)
    
    # there are not targets for every single nucleotide
    loss = loss_fn(stability_predicted[:, :labels.shape[1]], labels)

    return loss

def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load checkpoint for fine-tuning"""

    print("Resuming checkpoint", checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)

    model.bert.load_state_dict(checkpoint["model_bert_state_dict"])
    model.classifier.load_state_dict(checkpoint["model_classifier_state_dict"])
   
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    starting_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    loss_at_step = checkpoint.get("loss_at_step", [])

    return starting_epoch, global_step, train_losses, val_losses, loss_at_step

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
        model_bert_state_dict=model.bert.state_dict(),
        model_classifier_state_dict=model.classifier.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        train_losses=train_losses,
        val_losses=val_losses,
        loss_at_step=loss_at_step
    )
    torch.save(checkpoint, str(checkpoint_dir / f"{global_step}.pth"))


def finetune(model,
             train_dataloader,
             val_dataloader,
             output_dir,
             epochs=30000,
             lr=0.0001,
             val_interval_per_step=5,
             checkpoint_dir=None,
             checkpoint_interval=5):
    """Wrapper around training loop for finetuning stage"""

    train(
        model,
        train_dataloader,
        val_dataloader,
        output_dir,
        MSELoss(),
        calc_loss,
        load_checkpoint,
        save_checkpoint,
        checkpoint_dir,
        epochs=epochs,
        lr=lr,
        val_interval_per_step=val_interval_per_step,
        checkpoint_interval=checkpoint_interval)