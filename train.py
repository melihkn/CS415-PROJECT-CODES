import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

# Modeller
from models.HFANet.hfanet import HFANet, HFANet_timm
from models.HDANet.hdanet import HDANet
from models.stanet import STANet
from models.snunet import SNUNet_ECAM

# Dataloader
from data.dataset import get_dataloader
from utils.DiceLoss import DiceLoss
from helpers import train_one_epoch, validate



def calculate_iou(outputs, labels, threshold=0.5):
    # Model çıktısı (logits) -> Sigmoid -> 0 veya 1 (Binary Mask)
    # outputs: (Batch, 1, H, W)
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    
    # Düzleştirme (Flatten) - Piksel piksel karşılaştırma için
    outputs = outputs.view(-1)
    labels = labels.view(-1)
    
    # Kesişim ve Birleşim
    intersection = (outputs * labels).sum()
    union = outputs.sum() + labels.sum() - intersection
    
    # 0'a bölünme hatasını önlemek için küçük bir sayı (epsilon) ekliyoruz
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def train(args):
    # 1. Cihaz Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Modeli Başlatma
    print(f"Initializing model: {args.model}...")
    
    if args.model == 'hfanet':
        model = HFANet(encoder_name=args.backbone, classes=1, pretrained='imagenet')
    elif args.model == 'hfanet_timm':
        model = HFANet_timm(encoder_name=args.backbone, classes=1, pretrained=True)
    elif args.model == "hdanet":
        model = HDANet(n_classes=1, pretrained=True)
    elif args.model == "stanet":
        model = STANet(backbone_name=args.backbone, classes=1, pretrained=True)
    elif args.model == 'snunet':
        # SNUNet_ECAM: out_ch=1 (Binary Change Detection)
        # Deep Supervision kullanır, birden fazla çıktı üretir.
        model = SNUNet_ECAM(in_ch=3, out_ch=1) 
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Model to device
    model.to(device)
    print(f"Model {args.model} initialized with backbone {args.backbone}.")

    print(f"Loading data from {args.data_dir}...")
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split="train")
    val_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split="val")
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Change Detection Models')
    parser.add_argument('--model', type=str, default='snunet', 
                        choices=['hfanet', 'hfanet_timm', 'hdanet', 'stanet', 'snunet'], 
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Backbone encoder name')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (SNUNet uses 1e-3 or 1e-4)')
    
    args = parser.parse_args()
    train(args)