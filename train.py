import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Data
from data.dataset import get_dataloader

# Utils
from utils.losses import DiceLoss, FocalLoss, SoftIoULoss
from utils.training import train_one_epoch, CombinedLoss
from utils.evaluation import validate


def get_model(args, device):
    """
    Factory function to create model based on arguments.
    
    Supported models: snunet, hdanet, hfanet, hfanet_timm, stanet
    """
    model_name = args.model.lower()
    
    if model_name == 'snunet':
        from models.snunet import SNUNet_ECAM
        model = SNUNet_ECAM(in_ch=3, out_ch=1)
        print(f"Model: SNUNet-ECAM")
        
    elif model_name == 'hdanet':
        from models.HDANet.hdanet import HDANet
        model = HDANet(
            n_channels=3,
            n_classes=1,
            pretrained=True,
            ssl4eo_weights=args.ssl4eo_weights,
            ssl_method=args.ssl_method
        )
        print(f"Model: HDANet")
        
    elif model_name == 'hfanet':
        from models.HFANet.hfanet import HFANet
        model = HFANet(
            encoder_name=args.backbone,
            classes=1,
            pretrained='imagenet' if args.ssl4eo_weights is None else None,
            ssl4eo_weights=args.ssl4eo_weights,
            ssl_method=args.ssl_method
        )
        print(f"Model: HFANet with backbone={args.backbone}")
        
    elif model_name == 'hfanet_timm':
        from models.HFANet.hfanet import HFANet_timm
        model = HFANet_timm(encoder_name=args.backbone, classes=1, pretrained=True)
        print(f"Model: HFANet-TIMM with backbone={args.backbone}")
        
    elif model_name == 'stanet':
        from models.stanet import STANet
        model = STANet(backbone_name=args.backbone, classes=1, pretrained=True)
        print(f"Model: STANet with backbone={args.backbone}")
        
    else:
        raise ValueError(f"Unknown model: {args.model}. "
                        f"Choose from: snunet, hdanet, hfanet, hfanet_timm, stanet")
    
    return model.to(device)


def get_criterion(args):
    """
    Factory function to create loss function based on arguments.
    
    Supported losses: bce, dice, focal, softiou, bce+dice, bce+focal
    """
    loss_name = args.loss.lower()
    
    if loss_name == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        
    elif loss_name == 'dice':
        criterion = DiceLoss()
        
    elif loss_name == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
    elif loss_name == 'softiou':
        criterion = SoftIoULoss()
        
    elif loss_name == 'bce+dice':
        criterion = CombinedLoss([
            (nn.BCEWithLogitsLoss(), 1.0),
            (DiceLoss(), 1.0),
        ])
        
    elif loss_name == 'bce+focal':
        criterion = CombinedLoss([
            (nn.BCEWithLogitsLoss(), 1.0),
            (FocalLoss(), 1.0),
        ])
        
    else:
        raise ValueError(f"Unknown loss: {args.loss}. "
                        f"Choose from: bce, dice, focal, softiou, bce+dice, bce+focal")
    
    print(f"Loss function: {loss_name}")
    return criterion


def get_optimizer(model, args):
    """
    Factory function to create optimizer based on arguments.
    
    Supported optimizers: adamw, adam, sgd
    """
    opt_name = args.optimizer.lower()
    
    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    elif opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                             weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}. "
                        f"Choose from: adamw, adam, sgd")
    
    print(f"Optimizer: {opt_name} (lr={args.lr}, weight_decay={args.weight_decay})")
    return optimizer


def get_scheduler(optimizer, args):
    """
    Factory function to create learning rate scheduler.
    
    Supported schedulers: plateau, cosine, step, none
    """
    sched_name = args.scheduler.lower()
    
    if sched_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        
    elif sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
        
    elif sched_name == 'none':
        scheduler = None
        
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}. "
                        f"Choose from: plateau, cosine, step, none")
    
    if scheduler:
        print(f"Scheduler: {sched_name}")
    return scheduler


def train(args):
    """
    Main training function.
    
    Compatible with: SNUNet, HDANet, HFANet, HFANet-TIMM, STANet
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Create model
    model = get_model(args, device)
    
    # Create data loaders
    print(f"\nLoading data from: {args.data_dir}")
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Model:       {args.model}")
    print(f"  Backbone:    {args.backbone}")
    if args.ssl4eo_weights:
        print(f"  Pretrained:  SSL4EO-S12 ({args.ssl_method})")
    else:
        print(f"  Pretrained:  ImageNet")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print("=" * 60)
    train_loader = get_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split='train', 
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split='val', 
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create loss, optimizer, scheduler
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training state
    best_val_iou = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        train_loss, train_iou, train_f1 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            threshold=args.threshold
        )
        
        # Validate
        val_loss, val_iou, val_f1 = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold
        )
        
        # Print epoch summary
        print(f"\nEpoch [{epoch}/{args.epochs}] Summary:")
        print(f"  Train - Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_iou)
            else:
                scheduler.step()
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, f"best_{args.model}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_f1': val_f1,
                'args': vars(args),
            }, save_path)
            print(f"  ✓ New best model saved! (Val IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
        
        print()
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, f"final_{args.model}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'val_f1': val_f1,
        'args': vars(args),
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, f'best_{args.model}.pth')}")
    print(f"Final model saved to: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Change Detection Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='snunet',
                        choices=['snunet', 'hdanet', 'hfanet', 'hfanet_timm', 'stanet'],
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        help='Backbone encoder (for hfanet, hfanet_timm, stanet)')
    parser.add_argument('--ssl4eo_weights', type=str, default=None,
                        help='Path to SSL4EO-S12 weights (use with hfanet)')
    parser.add_argument('--ssl_method', type=str, default='moco', choices=['moco', 'dino'])

    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold for metrics')
    
    # Loss and optimizer
    parser.add_argument('--loss', type=str, default='bce+dice',
                        choices=['bce', 'dice', 'focal', 'softiou', 'bce+dice', 'bce+focal'],
                        help='Loss function')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)