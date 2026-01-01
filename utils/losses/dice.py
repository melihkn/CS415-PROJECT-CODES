import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Binary Dice Loss for segmentation tasks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
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
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()