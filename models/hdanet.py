import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .layers import ASPP

class DifferenceAttentionModule(nn.Module):
    """
    The 'Difference attention' block from the diagram.
    It learns a mask to apply to the absolute difference.
    """
    def __init__(self, in_channels):
        super(DifferenceAttentionModule, self).__init__()
        # A simple gate to learn the attention mask
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f_t1, f_t2):
        f_concat = torch.cat([f_t1, f_t2], dim=1)
        f_diff = torch.abs(f_t1 - f_t2)
        
        attention_mask = self.attention_gate(f_concat)
        
        # Refine the difference by multiplying it with the learned mask
        return f_diff * attention_mask


class HDANet_Head(nn.Module):
    """
    The full head of the HDANet, which corresponds to the 
    'ASPP', 'Difference attention', and 'Upsample' blocks in the diagram.
    """
    def __init__(self, in_channels_list, out_channels=256, num_classes=1):
        super(HDANet_Head, self).__init__()
        
        # Create ASPP and DAM for each feature level
        self.aspp_modules = nn.ModuleList()
        self.dam_modules = nn.ModuleList()
        
        for in_c in in_channels_list:
            # As per diagram, ASPP is applied to each branch's features
            # We'll have ASPP output a consistent 'out_channels'
            self.aspp_modules.append(ASPP(in_c, out_channels))
            # The DAM will then process these 256-channel features
            self.dam_modules.append(DifferenceAttentionModule(out_channels))
            
        # Final fusion and upsampling layers
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.final_classifier = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, t1_features, t2_features):
        # t1_features and t2_features are lists of tensors from the HRNet backbone
        
        refined_diff_maps = []
        # Target size is the largest feature map (the first one)
        target_size = t1_features[0].shape[2:]
        
        for i in range(len(t1_features)):
            f_t1 = self.aspp_modules[i](t1_features[i])
            f_t2 = self.aspp_modules[i](t2_features[i]) # Using same ASPP module
            
            diff_map = self.dam_modules[i](f_t1, f_t2)
            
            # Upsample all difference maps to the same size for concatenation
            diff_map_upsampled = F.interpolate(diff_map, size=target_size, mode='bilinear', align_corners=False)
            refined_diff_maps.append(diff_map_upsampled)
        
        # Concatenate all refined, upsampled maps
        fused_diff = torch.cat(refined_diff_maps, dim=1)
        
        fused_diff = self.fuse_conv(fused_diff)
        
        # Final upsample to original image size (HRNet's first stage is 1/4)
        out = F.interpolate(fused_diff, scale_factor=4, mode='bilinear', align_corners=False)
        out = self.final_classifier(out)
        
        return out

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
