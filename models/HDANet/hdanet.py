import torch.nn as nn
import torch.nn.functional as F
import timm
from .components import HDANet_Head


class HDANet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        super(HDANet, self).__init__()
        
        # 1. Load the HRNet backbone
        self.backbone = timm.create_model(
            'hrnet_w18',
            pretrained=pretrained,
            features_only=True,
            in_chans=n_channels
        )
        
        # Get the feature info
        feature_channels = self.backbone.feature_info.channels()
        
        # 2. Create the custom HDANet Head
        self.head = HDANet_Head(
            in_channels_list=feature_channels,
            out_channels=256, # Internal processing dim
            num_classes=n_classes
        )

    def forward(self, x1, x2):
        # 1. Pass T1 through the SHARED backbone
        t1_features = self.backbone(x1)
        
        # 2. Pass T2 through the *EXACT SAME* backbone
        t2_features = self.backbone(x2)
        
        # 3. Pass both sets of features to our custom head
        return self.head(t1_features, t2_features)
