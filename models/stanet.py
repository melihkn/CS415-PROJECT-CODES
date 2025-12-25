import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ResNetFPNFeatureExtractor, PAM

class STANet(nn.Module):
    def __init__(self, backbone_name='resnet34', fpn_out_channels=256, 
                 pam_channels=256, classes=1, pretrained=True, in_channels=3):
        super().__init__()
        
        # --- 1. Siamese Feature Extractor ---
        # Create ONE shared instance
        self.feature_extractor = ResNetFPNFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_channels=fpn_out_channels,
            in_channels=in_channels
        )
        
        # --- 2. Spatial-Temporal Attention Module ---
        # Using PAM here as it's more advanced than BAM alone
        # Input channels should match the output of the feature extractor
        self.pam = PAM(in_channels=fpn_out_channels) 
        
        # --- 3. Metric Module (Final Classifier Head) ---
        # Takes the PAM output and produces the final change map
        self.metric_module = nn.Sequential(
            nn.Conv2d(pam_channels, pam_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pam_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), # Optional dropout
            nn.Conv2d(pam_channels // 4, classes, kernel_size=1)
            # Final activation (Sigmoid for binary change) will be applied outside or in loss
        )

    def forward(self, x1, x2):
        # --- 1. Siamese Feature Extraction ---
        # Pass both images through the SHARED feature extractor
        feat_t1 = self.feature_extractor(x1) # X(1) in diagram
        feat_t2 = self.feature_extractor(x2) # X(2) in diagram
        
        # --- 2. Spatial-Temporal Attention ---
        # Apply PAM to both feature maps separately? Or concatenated?
        # The diagram seems to show applying attention *after* some implicit fusion?
        # Let's assume the standard way: apply attention to refine each map
        # *then* compare. OR, apply PAM to the *difference* or *concatenation*.
        # The diagram shows PAM applied to *both* X(1) and X(2) leading to Z(1) and Z(2).
        
        z1 = self.pam(feat_t1) # Z(1) in diagram
        z2 = self.pam(feat_t2) # Z(2) in diagram

        # --- 3. Final Comparison (Implicit in Metric Module) ---
        # How are Z(1) and Z(2) compared? Most likely subtraction or concatenation
        # Let's use subtraction as it's common for change detection
        feat_diff = torch.abs(z1 - z2)
        
        # --- 4. Metric Module (Classifier) ---
        change_map_logits = self.metric_module(feat_diff)
        
        # --- 5. Upsample to Original Size ---
        # The features are at 1/4 resolution, need to upsample
        change_map_logits = F.interpolate(change_map_logits, scale_factor=4, 
                                          mode='bilinear', align_corners=False)
                                          
        return change_map_logits
