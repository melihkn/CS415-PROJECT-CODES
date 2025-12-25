import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

# Modeller
from models.HFANet.hfanet import HFANet, HFANet_timm
from models.HDANet.hdanet import HDANet
from models.stanet import STANet

# Dataloader
from data.dataset import get_dataloader
from utils.DiceLoss import DiceLoss
from helpers import train_one_epoch, validate


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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Modellerin import edilmesi
from models.hfanet import HFANet, HFANet_timm
from models.HDANet.hdanet import HDANet
from models.stanet import STANet
from models.snunet import SNUNet_ECAM

# Dataset fonksiyonunu import ediyoruz
from data.dataset import get_dataloader 


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

    
    model.to(device)
    print(f"Model {args.model} initialized successfully.")

    # 3. Veri Yükleme (Dataset)
    print("Loading datasets...")
    # Train loader: Shuffle=True, Augmentation=True
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='train', img_size=256)
    
    # Val loader: Shuffle=False, Augmentation=False
    val_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='val', img_size=256)
    
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 4. Optimizer ve Loss
    # SNUNet için genelde learning rate 1e-3 veya 1e-4 iyidir.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Binary Cross Entropy (Logits ile): Sayısal olarak daha stabildir.
    criterion = nn.BCEWithLogitsLoss() 

    # En iyi modeli takip etmek için değişken
    best_val_loss = float('inf')
    best_val_iou = 0.0

    # 5. Eğitim Döngüsü
    print(f"Starting training loop for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_iou = 0.0  # YENİ: Toplam IoU değişkeni
        
        for i, batch in enumerate(train_loader):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device).float()

            optimizer.zero_grad()
            outputs = model(img_A, img_B)
            
            # Loss Hesaplama (Değişmedi)
            loss = 0
            if isinstance(outputs, (list, tuple)):
                # Deep Supervision: Tüm çıktıların loss'unu topla
                for output in outputs:
                    loss += criterion(output, label)
                
                # IoU için SADECE en son (en iyi) çıktıyı kullanıyoruz (genelde listenin ilk elemanı)
                final_output = outputs[0]
            else:
                loss = criterion(outputs, label)
                final_output = outputs

            loss.backward()
            optimizer.step()
            
            # İstatistikleri Güncelle
            train_loss += loss.item()
            train_iou += calculate_iou(final_output, label) # YENİ: Batch'in IoU'sunu ekle
        
        # Ortalamaları Al
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader) # YENİ
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0 # YENİ
        
        with torch.no_grad():
            for batch in val_loader:
                img_A = batch['image_A'].to(device)
                img_B = batch['image_B'].to(device)
                label = batch['label'].to(device).float()

                outputs = model(img_A, img_B)
                
                # Loss ve IoU Hesaplama
                batch_loss = 0
                if isinstance(outputs, (list, tuple)):
                    for output in outputs:
                        batch_loss += criterion(output, label)
                    final_output = outputs[0] # IoU için en iyi çıktı
                else:
                    batch_loss = criterion(outputs, label)
                    final_output = outputs
                
                val_loss += batch_loss.item()
                val_iou += calculate_iou(final_output, label) # YENİ

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader) # YENİ

        # EKRANA YAZDIRMA (GÜNCELLENDİ)
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")
        print(f"   Val   Loss: {avg_val_loss:.4f}   | Val   IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_val_iou:  # Loss küçüktür değil, IoU büyüktür kullanıyoruz
            best_val_iou = avg_val_iou
            
            # Klasör yoksa oluştur
            save_dir = 'checkpoints'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved to {save_path} (Val IoU: {best_val_iou:.4f})")

    print("Training finished.")

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