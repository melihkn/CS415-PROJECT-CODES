from .dice import DiceLoss
from .focal import FocalLoss
from .soft_iou import SoftIoULoss
from .denseCL import DenseContrastiveLoss

__all__ = ["DiceLoss", "FocalLoss", "SoftIoULoss", "DenseContastiveLoss"]