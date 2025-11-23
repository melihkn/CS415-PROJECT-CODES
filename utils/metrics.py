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
