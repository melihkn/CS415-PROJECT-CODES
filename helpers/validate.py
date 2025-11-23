import torch
from tqdm import tqdm
from utils.metrics import batch_metrics

def validate(model, dataloader, bce_loss, dice_loss, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            img_A = batch["image_A"].to(device)
            img_B = batch["image_B"].to(device)
            label = batch["label"].to(device)

            logits = model(img_A, img_B)

            loss_bce = bce_loss(logits, label.float())
            loss_dice = dice_loss(logits, label.float())
            loss = loss_bce + loss_dice

            running_loss += loss.item()

            metrics = batch_metrics(logits, label, threshold=threshold)
            running_iou += metrics["iou"]
            running_f1 += metrics["f1"]

            pbar.set_postfix({
                "val_loss": loss.item(),
                "val_IoU": metrics["iou"],
                "val_F1": metrics["f1"],
            })

    avg_loss = running_loss / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    avg_f1 = running_f1 / len(dataloader)

    return avg_loss, avg_iou, avg_f1