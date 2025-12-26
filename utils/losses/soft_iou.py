import torch
import torch.nn as nn

class SoftIoULoss(nn.Module):
    """
    Soft IoU (Jaccard) Loss for segmentation tasks.
    
    IoU = |A ∩ B| / |A ∪ B|
    Loss = 1 - IoU
    
    Uses soft probabilities instead of hard predictions for
    differentiability.
    
    Args:
        eps: Epsilon for numerical stability
        
    Compatible with: SNUNet, HDANet, HFANet, STANet, SegNet
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output [B, 1, H, W]
            targets: Binary ground truth [B, 1, H, W] with values 0/1
            
        Returns:
            loss: Scalar tensor
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        return 1 - iou.mean()