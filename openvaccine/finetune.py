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

    if "model_bert_state_dict" in checkpoint:
        print("Resuming checkpoint from finetuning step")
        model.bert.load_state_dict(checkpoint["model_bert_state_dict"])
        model.classifier.load_state_dict(checkpoint["model_classifier_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        loss_at_epoch = checkpoint.get("loss_at_epoch", [])
    else:
        print("Resuming checkpoint from pretraining")
        model.bert.load_state_dict(checkpoint["model_state_dict"])   
        epoch = 0
        global_step = 0
        train_losses = []
        val_losses = []
        loss_at_epoch = []

    return epoch, train_losses, val_losses, loss_at_epoch

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
        model_bert_state_dict=model.bert.state_dict(),
        model_classifier_state_dict=model.classifier.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        train_losses=train_losses,
        val_losses=val_losses,
        loss_at_epoch=loss_at_epoch
    )
    torch.save(checkpoint, str(checkpoint_dir / f"{epoch}.pth"))


def finetune(**train_args):
    """Wrapper around training loop for finetuning stage"""

    train(loss_fn=MSELoss(),
          calc_loss_fn=calc_loss,
          load_checkpoint_fn=load_checkpoint,
          save_checkpoint_fn=save_checkpoint,
          **train_args)