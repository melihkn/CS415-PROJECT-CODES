import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

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
    """
    print(f"Loading model weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print(f"  -> Full Checkpoint detected (Saved at Epoch: {checkpoint.get('epoch', 'Unknown')})")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
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
    print(f"Save Preds  : {'Enabled' if args.save_predictions else 'Disabled'}")

    # 2. Initialize Model
    model = SNUNet_ECAM(in_ch=3, out_ch=1, use_dense_cl=args.use_dense_cl).to(device)
    
    # 3. Load Weights
    load_checkpoint(model, args.model_path, device)
    model.eval()

    # 4. Load Test Data
    print(f"\nLoading dataset from: {args.data_dir} (Split: test)")
    test_dataset = ChangeDetectionDataset(
        root_dir=args.data_dir, 
        split='test', 
        img_size=args.img_size 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    print(f"Total test images: {len(test_dataset)}")

    # Prepare Output Directory for Predictions
    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        print(f"Predictions will be saved to: {pred_dir}")

    # Initialize counters
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0
    
    global_idx = 0 # To track image naming if 'name' is not in batch

    print("\nStarting inference loop...")
    print("-" * 50)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)
            
            # Bazı dataset yapılarında dosya ismi de döner, kontrol edelim
            names = batch.get('name', [])

            # Forward Pass
            outputs = model(img_A, img_B)
            
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            probs = torch.sigmoid(outputs)
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

            # --- Save Predictions if Enabled ---
            if args.save_predictions:
                # Convert predictions to numpy (0 or 255)
                preds_np = (preds.cpu().numpy() * 255).astype(np.uint8)
                
                for i in range(preds_np.shape[0]):
                    mask = preds_np[i, 0, :, :] # (H, W)
                    
                    # Dosya ismini belirle
                    if len(names) > i:
                        file_name = names[i]
                        # Uzantıyı garantile
                        if not file_name.lower().endswith('.png'):
                            file_name = os.path.splitext(file_name)[0] + '.png'
                    else:
                        file_name = f"pred_{global_idx}.png"
                    
                    save_path = os.path.join(pred_dir, file_name)
                    Image.fromarray(mask).save(save_path)
                    global_idx += 1

    # 5. Calculate Final Metrics
    precision, recall, f1, iou, acc = calculate_metrics(total_TP, total_FP, total_FN, total_TN)

    print("\n" + "="*50)
    print(f"       FINAL TEST RESULTS")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"IoU       : {iou:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print("-" * 50)
    if args.save_predictions:
        print(f"Predictions saved to: {pred_dir}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SNUNet-CD Model")
    
    # Path Arguments
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint file')
    
    # Output Arguments (EKSİK OLANLAR EKLENDİ)
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Directory to save outputs')
    parser.add_argument('--save_predictions', action='store_true', help='Flag to save predicted masks')
    
    # Model Arguments
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--use_dense_cl', action='store_true', help='Set this if model was trained with DenseCL')
    
    # Hyperparameters
    parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for binary classification')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')

    args = parser.parse_args()
    
    evaluate(args)