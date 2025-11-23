import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

# Modeller
from models.hfanet import HFANet, HFANet_timm
from models.HDANet.hdanet import HDANet
from models.stanet import STANet

# Dataloader
from data.dataset import get_dataloader
from utils.DiceLoss import DiceLoss
from helpers import train_one_epoch, validate, evaluate_on_loader


def train(args):
    """
    Train the model and test it on test set
    
    Args:
        args: command line arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    print(f"Loading data from {args.data_dir}...")
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split="train")
    val_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split="val")

    # Model selection
    if args.model == "hfanet":
        model = HFANet(encoder_name=args.backbone, classes=1, pretrained="imagenet")
    elif args.model == "hfanet_timm":
        model = HFANet_timm(encoder_name=args.backbone, classes=1, pretrained=True)
    elif args.model == "hdanet":
        model = HDANet(n_classes=1, pretrained=True)
    elif args.model == "stanet":
        model = STANet(backbone_name=args.backbone, classes=1, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Model to device
    model.to(device)
    print(f"Model {args.model} initialized with backbone {args.backbone}.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_iou = 0.0
    patience_counter = 0
    threshold = 0.5  # binarization

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou, train_f1 = train_one_epoch(
            model, train_loader, bce_loss, dice_loss, optimizer, device, epoch, threshold=threshold
        )
        val_loss, val_iou, val_f1 = validate(
            model, val_loader, bce_loss, dice_loss, device, threshold=threshold
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}"
        )

        # LR is adjusted based on IoU
        scheduler.step(val_iou)

        # Save best model based on IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            save_path = os.path.join("checkpoints", f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (Val IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
            print(f"No IoU improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building Change Detection Training")

    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--model", type=str, default="hfanet",
                        choices=["hfanet", "hfanet_timm", "hdanet", "stanet"])
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone name")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()
    train(args)
