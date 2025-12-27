import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .metrics import batch_metrics, iou_from_confusion, f1_from_confusion


def normalize_model_output(outputs):
    """
    Normalizes model output to single tensor.
    Handles both single output and deep supervision (tuple/list) cases.
    
    Args:
        outputs: Model output - either tensor or tuple/list of tensors
        
    Returns:
        logits: Single tensor [B, 1, H, W]
    """
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def compute_loss(criterion, outputs, targets):
    """
    Computes loss handling both single and multi-output models.
    
    For deep supervision models (SNUNet), loss is computed on all outputs.
    For single output models, loss is computed normally.
    Automatically resizes outputs to match target size if needed.
    
    Args:
        criterion: Loss function
        outputs: Model output (tensor or tuple/list)
        targets: Ground truth [B, 1, H, W]
        
    Returns:
        loss: Scalar tensor
    """
    target_size = targets.shape[-2:]  # (H, W)
    
    if isinstance(outputs, (list, tuple)):
        # Deep supervision - sum losses from all outputs
        total_loss = 0.0
        for output in outputs:
            # Resize output to match target if needed
            if output.shape[-2:] != target_size:
                output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
            total_loss += criterion(output, targets)
        return total_loss / len(outputs)  # Average loss
    
    # Resize output to match target if needed
    if outputs.shape[-2:] != target_size:
        outputs = F.interpolate(outputs, size=target_size, mode='bilinear', align_corners=False)
    return criterion(outputs, targets)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    threshold: float = 0.5
):
    """
    Trains the model for one epoch.
    
    Compatible with: SNUNet, HDANet, HFANet, HFANet-TIMM, STANet
    Handles both single output and deep supervision models.
    
    Args:
        model: Change detection model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer instance
        device: torch.device (cuda or cpu)
        epoch: Current epoch number
        threshold: Binarization threshold for metrics
        
    Returns:
        tuple: (epoch_loss, epoch_iou, epoch_f1)
    """
    model.train()
    
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    num_batches = 0
    running_tp = 0
    running_fp = 0
    running_fn = 0
    running_tn = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        img_A = batch["image_A"].to(device)
        img_B = batch["image_B"].to(device)
        label = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(img_A, img_B)
        
        # Compute loss (handles deep supervision)
        loss = compute_loss(criterion, outputs, label)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics (use normalized single output)
        running_loss += loss.item()
        num_batches += 1

        logits = normalize_model_output(outputs)
        # Resize logits to match label size if needed
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(logits, size=label.shape[-2:], mode='bilinear', align_corners=False)
        metrics = batch_metrics(logits, label, threshold=threshold)
        running_iou += metrics["iou"]
        running_f1 += metrics["f1"]
        running_tp += metrics["tp"]
        running_fp += metrics["fp"]
        running_fn += metrics["fn"]
        running_tn += metrics["tn"]

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "IoU": f"{metrics['iou']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
        })

    epoch_loss = running_loss / num_batches
    epoch_iou = iou_from_confusion(running_tp, running_fp, running_fn)
    epoch_f1 = f1_from_confusion(running_tp, running_fp, running_fn)
    
    # Handle both tensor and float returns
    if hasattr(epoch_iou, 'item'):
        epoch_iou = epoch_iou.item()
    if hasattr(epoch_f1, 'item'):
        epoch_f1 = epoch_f1.item()

    return epoch_loss, epoch_iou, epoch_f1


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with optional weights.
    
    Example:
        criterion = CombinedLoss([
            (DiceLoss(), 1.0),
            (nn.BCEWithLogitsLoss(), 1.0),
        ])
    """
    def __init__(self, losses_with_weights):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses_with_weights])
        self.weights = [weight for _, weight in losses_with_weights]
    
    def forward(self, logits, targets):
        # Resize logits to match target size if needed
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(logits, targets)
        return total_loss