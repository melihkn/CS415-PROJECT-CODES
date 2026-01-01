import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reduces the loss contribution from easy examples and focuses
    on hard negatives.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
        
    Compatible with: SNUNet, HDANet, HFANet, STANet, SegNet
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output [B, 1, H, W]
            targets: Binary ground truth [B, 1, H, W] with values 0/1
            
        Returns:
            loss: Scalar tensor (if reduction is 'mean' or 'sum')
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # p_t = probability of correct class
        p_t = torch.exp(-bce)
        focal_term = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_term * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss