import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.hfanet import HFANet, HFANet_timm
from models.HDANet.hdanet import HDANet
from models.stanet import STANet
# from data.dataset import get_dataloader # Uncomment when dataset is ready

def train(args):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Initialize Model
    if args.model == 'hfanet':
        model = HFANet(encoder_name=args.backbone, classes=1, pretrained='imagenet')
    elif args.model == 'hfanet_timm':
        model = HFANet_timm(encoder_name=args.backbone, classes=1, pretrained=True)
    elif args.model == 'hdanet':
        model = HDANet(n_classes=1, pretrained=True)
    elif args.model == 'stanet':
        model = STANet(backbone_name=args.backbone, classes=1, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.to(device)
    print(f"Model {args.model} initialized.")

    # 3. Setup Data (Placeholder)
    # train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='train')
    # val_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='val')
    print("Data loaders not initialized (Dataset path required).")

    # 4. Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Assuming binary change detection

    # 5. Training Loop (Template)
    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # for batch in train_loader:
        #     img_A = batch['image_A'].to(device)
        #     img_B = batch['image_B'].to(device)
        #     label = batch['label'].to(device)
        #
        #     optimizer.zero_grad()
        #     outputs = model(img_A, img_B)
        #     loss = criterion(outputs, label)
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {running_loss:.4f}")
        
        # Validation...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Change Detection Models')
    parser.add_argument('--model', type=str, default='hfanet', choices=['hfanet', 'hfanet_timm', 'hdanet', 'stanet'], help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Backbone encoder name')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    train(args)
