import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from models.snunet import SNUNet_ECAM
from data.dataset import ChangeDetectionDataset

def calculate_metrics(TP, FP, FN, TN):
    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps)
    return precision, recall, f1_score, iou, accuracy

def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading model weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("  -> Weights loaded successfully! ✅")

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEVALUATION CONFIGURATION")
    print(f"Device: {device}")
    
    # Model Init
    model = SNUNet_ECAM(in_ch=3, out_ch=1, use_dense_cl=args.use_dense_cl).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    # Data Loading
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

    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        print(f"Predictions will be saved to: {pred_dir}")

    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    
    print("\nStarting inference loop...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)
            
            # [KRİTİK] Dosya isimlerini batch'ten alıyoruz
            names = batch.get('name') 

            outputs = model(img_A, img_B)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            probs = torch.sigmoid(outputs)
            preds = (probs > args.threshold).float()

            # Metrics
            TP = ((preds == 1) & (label == 1)).sum().item()
            FP = ((preds == 1) & (label == 0)).sum().item()
            FN = ((preds == 0) & (label == 1)).sum().item()
            TN = ((preds == 0) & (label == 0)).sum().item()
            total_TP += TP; total_FP += FP; total_FN += FN; total_TN += TN

            # Saving
            if args.save_predictions:
                preds_np = (preds.cpu().numpy() * 255).astype(np.uint8)
                for i in range(preds_np.shape[0]):
                    mask = preds_np[i, 0, :, :]
                    
                    # Dosya ismini belirle
                    if names and len(names) > i:
                        file_name = names[i] # Orijinal isim (örn: test_1.png)
                    else:
                        file_name = f"unknown_{i}.png" # Yedek isim
                    
                    save_path = os.path.join(pred_dir, file_name)
                    Image.fromarray(mask).save(save_path)

    precision, recall, f1, iou, acc = calculate_metrics(total_TP, total_FP, total_FN, total_TN)
    print("\n" + "="*50)
    print(f"       FINAL TEST RESULTS")
    print("="*50)
    print(f"IoU       : {iou:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./test_results')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--use_dense_cl', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    evaluate(args)