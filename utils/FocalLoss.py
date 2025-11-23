import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss
    args:
        alpha: positive class weight
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
    returns:
        loss: scalar
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Binary Focal Loss
        args:
            logits: [B,1,H,W]
            targets: [B,1,H,W] 0/1
        returns:
            loss: scalar
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # p_t = exp(-BCE)  => doğru sınıf olasılığı
        p_t = torch.exp(-bce)
        focal_term = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_term * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss