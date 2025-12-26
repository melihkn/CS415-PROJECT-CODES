import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

from .metrics import (
    compute_confusion_matrix,
    iou_from_confusion,
    f1_from_confusion,
    precision_from_confusion,
    recall_from_confusion,
    sigmoid_to_probs,
    batch_metrics,
)
from .training import normalize_model_output, compute_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
):
    """
    Validates the model on a given dataloader.
    
    Compatible with: SNUNet, HDANet, HFANet, HFANet-TIMM, STANet
    Handles both single output and deep supervision models.
    
    Args:
        model: Change detection model
        dataloader: Validation dataloader
        criterion: Loss function
        device: torch.device
        threshold: Binarization threshold
        
    Returns:
        tuple: (avg_loss, avg_iou, avg_f1)
    """
    model.eval()
    
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            img_A = batch["image_A"].to(device)
            img_B = batch["image_B"].to(device)
            label = batch["label"].to(device)

            outputs = model(img_A, img_B)
            
            # Compute loss (handles deep supervision)
            loss = compute_loss(criterion, outputs, label)
            running_loss += loss.item()
            num_batches += 1

            # Compute metrics on normalized output
            logits = normalize_model_output(outputs)
            metrics = batch_metrics(logits, label, threshold=threshold)
            running_iou += metrics["iou"]
            running_f1 += metrics["f1"]

            pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}",
                "val_IoU": f"{metrics['iou']:.4f}",
                "val_F1": f"{metrics['f1']:.4f}",
            })

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_f1 = running_f1 / num_batches

    return avg_loss, avg_iou, avg_f1


def evaluate_on_loader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_results_path: str = None,
    save_pr_curve_path: str = None
):
    """
    Comprehensive evaluation with optional prediction saving and PR curve.
    
    Compatible with: SNUNet, HDANet, HFANet, HFANet-TIMM, STANet
    
    Args:
        model: Trained change detection model
        dataloader: Test dataloader
        device: torch.device
        threshold: Binarization threshold
        save_results_path: Directory to save predictions (optional)
        save_pr_curve_path: Path to save PR curve (optional)
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()

    total_tp = total_fp = total_fn = total_tn = 0.0
    all_probs = []
    all_targets = []

    if save_results_path is not None:
        os.makedirs(save_results_path, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating [Test]")
        for batch in pbar:
            img_A = batch["image_A"].to(device)
            img_B = batch["image_B"].to(device)
            label = batch["label"].to(device)
            filenames = batch["filename"]

            # Forward pass and normalize output
            outputs = model(img_A, img_B)
            logits = normalize_model_output(outputs)
            probs = sigmoid_to_probs(logits)

            # Confusion matrix
            tp, fp, fn, tn = compute_confusion_matrix(logits, label, threshold=threshold)
            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            total_tn += tn.item()

            # Collect for PR curve
            all_probs.append(probs.view(-1).cpu().numpy())
            all_targets.append(label.view(-1).cpu().numpy())

            # Save predictions
            if save_results_path is not None:
                _save_batch_predictions(probs, filenames, save_results_path, threshold)

            current_iou = iou_from_confusion(total_tp, total_fp, total_fn)
            pbar.set_postfix({"IoU": f"{current_iou:.4f}"})

    # Compute final metrics
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = _compute_final_metrics(
        total_tp, total_fp, total_fn, total_tn,
        all_probs, all_targets
    )

    _print_metrics(metrics)

    if save_pr_curve_path is not None:
        _save_pr_curve(all_targets, all_probs, metrics["ap"], save_pr_curve_path)

    return metrics


# ============================================================
# Private Helper Functions
# ============================================================

def _save_batch_predictions(probs, filenames, save_dir, threshold):
    """Saves each prediction in the batch as an image."""
    batch_size = probs.shape[0]
    
    for i in range(batch_size):
        pred = probs[i]
        pred_binary = (pred > threshold).float()
        pred_pil = TF.to_pil_image(pred_binary.cpu())
        save_path = os.path.join(save_dir, filenames[i])
        pred_pil.save(save_path)


def _compute_final_metrics(tp, fp, fn, tn, all_probs, all_targets):
    """Computes all final metrics including AP."""
    ap = average_precision_score(all_targets, all_probs)
    
    return {
        "iou": float(iou_from_confusion(tp, fp, fn)),
        "f1": float(f1_from_confusion(tp, fp, fn)),
        "precision": float(precision_from_confusion(tp, fp)),
        "recall": float(recall_from_confusion(tp, fn)),
        "ap": float(ap),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _print_metrics(metrics):
    """Prints evaluation metrics."""
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"  IoU        : {metrics['iou']:.4f}")
    print(f"  F1-score   : {metrics['f1']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  AP (PR AUC): {metrics['ap']:.4f}")
    print("-" * 50)
    print(f"  TP: {metrics['tp']:.0f}  |  FP: {metrics['fp']:.0f}")
    print(f"  FN: {metrics['fn']:.0f}  |  TN: {metrics['tn']:.0f}")
    print("=" * 50 + "\n")


def _save_pr_curve(targets, probs, ap, save_path):
    """Saves PR curve plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    pr_precision, pr_recall, _ = precision_recall_curve(targets, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(pr_recall, pr_precision, linewidth=2, color='blue')
    plt.fill_between(pr_recall, pr_precision, alpha=0.2, color='blue')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve (AP = {ap:.4f})", fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PR curve saved to: {save_path}")