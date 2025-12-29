import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports from your project structure
from models.snunet import SNUNet_ECAM
from data.dataset import ChangeDetectionDataset

def calculate_metrics(TP, FP, FN, TN):
    """
    Calculates metrics based on Confusion Matrix values.
    """
    eps = 1e-6
    
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps)
    
    return precision, recall, f1_score, iou, accuracy

def load_checkpoint(model, checkpoint_path, device):
    """
    Smartly loads the checkpoint file.
    Handles both full training checkpoints (with optimizer states) and weight-only files.
    """
    print(f"Loading model weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if checkpoint is a dictionary containing metadata (epoch, optimizer, etc.)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print(f"  -> Full Checkpoint detected (Saved at Epoch: {checkpoint.get('epoch', 'Unknown')})")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it only contains weights (state_dict)
        print("  -> State Dict (Weights Only) detected.")
        model.load_state_dict(checkpoint)
    
    print("  -> Weights loaded successfully! ✅")

def evaluate(args):
    # 1. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*50}")
    print(f"Device      : {device}")
    print(f"Image Size  : {args.img_size}x{args.img_size}")
    print(f"Threshold   : {args.threshold}")
    print(f"DenseCL Mode: {'Enabled' if args.use_dense_cl else 'Disabled'}")

    # 2. Initialize Model
    # CRITICAL: If trained with DenseCL, 'use_dense_cl' must be True to match architecture keys.
    model = SNUNet_ECAM(in_ch=3, out_ch=1, use_dense_cl=args.use_dense_cl).to(device)
    
    # 3. Load Weights
    load_checkpoint(model, args.model_path, device)
    model.eval()

    # 4. Load Test Data
    print(f"\nLoading dataset from: {args.data_dir} (Split: test)")
    test_dataset = ChangeDetectionDataset(
        root_dir=args.data_dir, 
        split='test', 
        img_size=args.img_size # Must match training size (e.g., 512)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    print(f"Total test images: {len(test_dataset)}")

    # Initialize counters for Confusion Matrix
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    print("\nStarting inference loop...")
    print("-" * 50)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)

            # Forward Pass
            outputs = model(img_A, img_B)
            
            # SNUNet returns a list of outputs [out1, out2, out3, out4].
            # We use the first one (deepest supervision) or combine them.
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Apply Sigmoid to get probabilities (0-1)
            probs = torch.sigmoid(outputs)
            
            # Apply Binary Threshold (Default 0.4)
            preds = (probs > args.threshold).float()

            # --- Calculate Pixel-wise Statistics ---
            TP = ((preds == 1) & (label == 1)).sum().item()
            FP = ((preds == 1) & (label == 0)).sum().item()
            FN = ((preds == 0) & (label == 1)).sum().item()
            TN = ((preds == 0) & (label == 0)).sum().item()

            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

    # 5. Calculate Final Metrics
    precision, recall, f1, iou, acc = calculate_metrics(total_TP, total_FP, total_FN, total_TN)

    print("\n" + "="*50)
    print(f"       FINAL TEST RESULTS")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"IoU       : {iou:.4f}  <-- (Intersection over Union)")
    print(f"Accuracy  : {acc:.4f}")
    print("-" * 50)
    print(f"Confusion Matrix Counts:")
    print(f"TP: {int(total_TP)} | FP: {int(total_FP)}")
    print(f"FN: {int(total_FN)} | TN: {int(total_TN)}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SNUNet-CD Model")
    
    # Path Arguments
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint file')
    
    # Model Arguments
    parser.add_argument('--img_size', type=int, default=512, help='Input image size (must match training)')
    parser.add_argument('--use_dense_cl', action='store_true', help='Set this if model was trained with DenseCL')
    
    # Hyperparameters
    parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for binary classification (0.0 - 1.0)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')

    args = parser.parse_args()
    
    evaluate(args)