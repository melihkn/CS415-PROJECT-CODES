import argparse
import os
import json
import torch

# Models
from models.snunet import SNUNet_ECAM
from models.stanet import STANet
from models.HDANet.hdanet import HDANet
from models.HFANet.hfanet import HFANet, HFANet_timm

# Data
from data.dataset import get_dataloader

# Utils
from utils.evaluation import evaluate_on_loader


def get_model(model_name, backbone='resnet34'):
    """
    Factory function to create model based on name.
    
    Supported models: snunet, hdanet, hfanet, hfanet_timm, stanet
    """
    model_name = model_name.lower()
    
    if model_name == 'snunet':
        model = SNUNet_ECAM(in_ch=3, out_ch=1)
        
    elif model_name == 'hdanet':
        model = HDANet(n_classes=1, pretrained=False)
        
    elif model_name == 'hfanet':
        model = HFANet(encoder_name=backbone, classes=1, pretrained=None)
        
    elif model_name == 'hfanet_timm':
        model = HFANet_timm(encoder_name=backbone, classes=1, pretrained=False)
        
    elif model_name == 'stanet':
        model = STANet(backbone_name=backbone, classes=1, pretrained=False)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: snunet, hdanet, hfanet, hfanet_timm, stanet")
    
    return model


def load_checkpoint(model, checkpoint_path, device):
    """
    Loads model weights from checkpoint.
    
    Supports both full checkpoint (with optimizer, epoch, etc.) 
    and state_dict only formats.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val IoU: {checkpoint.get('val_iou', 'N/A')}")
        
        # Return training args if available
        return checkpoint.get('args', {})
    else:
        # Legacy format - just state_dict
        model.load_state_dict(checkpoint)
        return {}


def test(args):
    """
    Main testing function.
    
    Compatible with: SNUNet, HDANet, HFANet, HFANet-TIMM, STANet
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Create model
    model = get_model(args.model, args.backbone)
    model = model.to(device)
    
    # Load checkpoint
    saved_args = load_checkpoint(model, args.checkpoint, device)
    
    # Use saved backbone if available and not overridden
    if saved_args and 'backbone' in saved_args and args.backbone == 'resnet34':
        print(f"  Using saved backbone: {saved_args['backbone']}")
    
    # Create test dataloader
    print(f"\nLoading test data from: {args.data_dir}")
    test_loader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        split='test',
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    print(f"Test batches: {len(test_loader)}")
    
    # Create output directories
    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)
        save_results_path = os.path.join(args.output_dir, 'predictions')
    else:
        save_results_path = None
    
    if args.save_pr_curve:
        pr_curve_path = os.path.join(args.output_dir, 'pr_curve.png')
    else:
        pr_curve_path = None
    
    print(f"\n{'='*60}")
    print(f"STARTING EVALUATION")
    print(f"{'='*60}\n")
    
    # Run evaluation
    metrics = evaluate_on_loader(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
        save_results_path=save_results_path,
        save_pr_curve_path=pr_curve_path
    )
    
    # Save metrics to JSON
    if args.save_metrics:
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Change Detection Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['snunet', 'hdanet', 'hfanet', 'hfanet_timm', 'stanet'],
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        help='Backbone encoder (for hfanet, hfanet_timm, stanet)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold for predictions')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction images')
    parser.add_argument('--save_pr_curve', action='store_true',
                        help='Save precision-recall curve')
    parser.add_argument('--save_metrics', action='store_true', default=True,
                        help='Save metrics to JSON file')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)