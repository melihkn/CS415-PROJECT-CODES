import torch


# ============================================================
# Utility Functions
# ============================================================

def sigmoid_to_probs(logits):
    """Converts raw logits to probabilities using sigmoid."""
    return torch.sigmoid(logits)


def binarize_probs(probs, threshold=0.5):
    """Converts probabilities to binary predictions."""
    return (probs > threshold).float()


# ============================================================
# Confusion Matrix Based Metrics
# ============================================================

def compute_confusion_matrix(logits, targets, threshold=0.5):
    """
    Computes confusion matrix components (TP, FP, FN, TN).
    
    Args:
        logits: Raw model output [B, 1, H, W]
        targets: Binary ground truth [B, 1, H, W] with values 0/1
        threshold: Binarization threshold
        
    Returns:
        tuple: (tp, fp, fn, tn) as tensors
    """
    probs = sigmoid_to_probs(logits)
    preds = binarize_probs(probs, threshold=threshold)
    targets = targets.float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    return tp, fp, fn, tn


def iou_from_confusion(tp, fp, fn, eps=1e-6):
    """Computes IoU from confusion matrix components."""
    return (tp + eps) / (tp + fp + fn + eps)


def f1_from_confusion(tp, fp, fn, eps=1e-6):
    """Computes F1 score from confusion matrix components."""
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def precision_from_confusion(tp, fp, eps=1e-6):
    """Computes precision from confusion matrix components."""
    return (tp + eps) / (tp + fp + eps)


def recall_from_confusion(tp, fn, eps=1e-6):
    """Computes recall from confusion matrix components."""
    return (tp + eps) / (tp + fn + eps)


def accuracy_from_confusion(tp, fp, fn, tn, eps=1e-6):
    """Computes accuracy from confusion matrix components."""
    return (tp + tn + eps) / (tp + fp + fn + tn + eps)


# ============================================================
# Direct Metric Calculations
# ============================================================

def calculate_iou(logits, labels, threshold=0.5, eps=1e-6):
    """
    Calculates Intersection over Union (IoU) for a batch.
    
    Args:
        logits: Raw model output [B, 1, H, W]
        labels: Binary ground truth [B, 1, H, W] with values 0/1
        threshold: Binarization threshold
        eps: Epsilon for numerical stability
        
    Returns:
        iou: Scalar tensor (batch mean)
    """
    probs = torch.sigmoid(logits)
    preds_bin = (probs > threshold).float()
    labels = labels.float()

    intersection = (preds_bin * labels).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def calculate_f1(logits, labels, threshold=0.5, eps=1e-6):
    """
    Calculates F1 score for a batch.
    
    Args:
        logits: Raw model output [B, 1, H, W]
        labels: Binary ground truth [B, 1, H, W] with values 0/1
        threshold: Binarization threshold
        eps: Epsilon for numerical stability
        
    Returns:
        f1: Scalar tensor (batch mean)
    """
    probs = torch.sigmoid(logits)
    preds_bin = (probs > threshold).float()
    labels = labels.float()

    tp = (preds_bin * labels).sum(dim=(1, 2, 3))
    fp = (preds_bin * (1 - labels)).sum(dim=(1, 2, 3))
    fn = ((1 - preds_bin) * labels).sum(dim=(1, 2, 3))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean()


# ============================================================
# Batch Metrics (All-in-One)
# ============================================================

def batch_metrics(logits, targets, threshold=0.5, eps=1e-6):
    """
    Computes all metrics for a batch in one pass.
    
    Args:
        logits: Raw model output [B, 1, H, W]
        targets: Binary ground truth [B, 1, H, W] with values 0/1
        threshold: Binarization threshold
        eps: Epsilon for numerical stability
        
    Returns:
        dict: Dictionary containing iou, f1, precision, recall, tp, fp, fn, tn
    """
    tp, fp, fn, tn = compute_confusion_matrix(logits, targets, threshold)

    iou = iou_from_confusion(tp, fp, fn, eps)
    f1 = f1_from_confusion(tp, fp, fn, eps)
    precision = precision_from_confusion(tp, fp, eps)
    recall = recall_from_confusion(tp, fn, eps)
    accuracy = accuracy_from_confusion(tp, fp, fn, tn, eps)

    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "accuracy": accuracy.item(),
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
    }