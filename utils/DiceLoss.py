import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Binary Dice Loss
    args:
        eps: epsilon for numerical stability
    returns:
        loss: scalar
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        Binary Dice Loss
        args:
            logits: [B,1,H,W]
            targets: [B,1,H,W] 0/1
        returns:
            loss: scalar
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()