import torch
from tqdm import tqdm
from utils.metrics import batch_metrics
import torch.nn as nn
import torch.optim as optim
from utils.DiceLoss import DiceLoss




def train_one_epoch(model : torch.nn.Module, dataloader : torch.utils.data.DataLoader, bce_loss : nn.BCEWithLogitsLoss, dice_loss : DiceLoss, optimizer : optim.AdamW, device : torch.device, epoch : int, threshold=0.5):
    """
    One epoch training

    args:
        model : torch.nn.Module
        dataloader : torch.utils.data.DataLoader
        bce_loss : nn.BCEWithLogitsLoss
        dice_loss : DiceLoss
        optimizer : optim.AdamW
        device : torch.device
        epoch : int
        threshold : float

    returns:
        epoch_loss : float
        epoch_iou : float
        epoch_f1 : float
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch in pbar:
        img_A = batch["image_A"].to(device)
        img_B = batch["image_B"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(img_A, img_B)

        loss_bce = bce_loss(logits, label.float())
        loss_dice = dice_loss(logits, label.float())
        loss = loss_bce + loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        metrics = batch_metrics(logits, label, threshold=threshold)
        running_iou += metrics["iou"]
        running_f1 += metrics["f1"]

        pbar.set_postfix({
            "loss": loss.item(),
            "IoU": metrics["iou"],
            "F1": metrics["f1"],
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_f1 = running_f1 / len(dataloader)

    return epoch_loss, epoch_iou, epoch_f1