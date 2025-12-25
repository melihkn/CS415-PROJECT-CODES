import torch

def calculate_iou(preds, labels, eps=1e-6):
    """
    Calculate Intersection over Union (IoU) for a batch.
    args:
        preds: logits [B,1,H,W]
        labels: [B,1,H,W] 0/1
        eps: epsilon for numerical stability
    returns:
        iou: scalar
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).float()
    labels = labels.float()

    intersection = (preds_bin * labels).sum(dim=(1,2,3))
    union = preds_bin.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()  # batch mean IoU

def calculate_iou2(outputs, labels, threshold=0.5):
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

def calculate_f1(preds, labels, eps=1e-6):
    """
    Calculate F1 score for a batch.
    args:
        preds: logits [B,1,H,W]
        labels: [B,1,H,W] 0/1
        eps: epsilon for numerical stability
    returns:
        f1: scalar
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).float()
    labels = labels.float()

    tp = (preds_bin * labels).sum(dim=(1,2,3))
    fp = (preds_bin * (1 - labels)).sum(dim=(1,2,3))
    fn = ((1 - preds_bin) * labels).sum(dim=(1,2,3))

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean()


# metrics second approach

def sigmoid_to_probs(logits):
    return torch.sigmoid(logits)


def binarize_probs(probs, threshold=0.5):
    return (probs > threshold).float()


def compute_confusion_matrix(logits, targets, threshold=0.5, eps=1e-6):
    """
    logits: [B,1,H,W]
    targets: [B,1,H,W] (0/1)
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
    return (tp + eps) / (tp + fp + fn + eps)


def f1_from_confusion(tp, fp, fn, eps=1e-6):
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def precision_from_confusion(tp, fp, eps=1e-6):
    return (tp + eps) / (tp + fp + eps)


def recall_from_confusion(tp, fn, eps=1e-6):
    return (tp + eps) / (tp + fn + eps)


def batch_metrics(logits, targets, threshold=0.5, eps=1e-6):
    tp, fp, fn, tn = compute_confusion_matrix(logits, targets, threshold, eps)

    iou = iou_from_confusion(tp, fp, fn, eps)
    f1 = f1_from_confusion(tp, fp, fn, eps)
    precision = precision_from_confusion(tp, fp, eps)
    recall = recall_from_confusion(tp, fn, eps)

    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
    }
