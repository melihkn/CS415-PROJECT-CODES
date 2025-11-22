import torch.nn as nn
from .difference_attention_module import DifferenceAttentionModule
from .aspp import ASPP
import torch.nn.functional as F
import torch

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