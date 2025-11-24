import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm





class BAM(nn.Module):
    """ Implements the Basic Attention Module (BAM). """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # 1x1 convolutions to generate Query, Key, Value
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1) # Apply softmax along the key dimension

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate Q, K, V
        # Reshape for matrix multiplication: (B, C', N) where N = H*W
        proj_query = self.query_conv(x).view(batch_size, -1, W * H).permute(0, 2, 1) # B, N, C'
        proj_key = self.key_conv(x).view(batch_size, -1, W * H) # B, C', N
        proj_value = self.value_conv(x).view(batch_size, -1, W * H) # B, C, N
        
        # Calculate attention map: A = softmax(Q @ K)
        energy = torch.bmm(proj_query, proj_key) # B, N, N
        attention = self.softmax(energy) # B, N, N
        
        # Apply attention to Value: out = V @ A.T
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, N
        
        # Reshape back to image format: (B, C, H, W)
        out = out.view(batch_size, C, H, W)
        
        # Apply learnable scaling and add residual connection
        out = self.gamma * out + x
        return out

class PAM(nn.Module):
    """ Implements the Pyramid Attention Module (PAM). """
    def __init__(self, in_channels):
        super().__init__()
        
        # Define scales for the pyramid (e.g., 1x1, 2x2, 3x3, 6x6 avg pooling)
        self.scales = [1, 2, 3, 6] 
        
        # Create BAM modules for each scale's processing path
        self.bams = nn.ModuleList()
        # Create 1x1 convs to reduce channels before BAM at different scales
        self.convs = nn.ModuleList()
        
        for _ in self.scales:
            # Reduce channels slightly before BAM? (Optional, helps efficiency)
            # Or keep the same channels as input
            processed_channels = in_channels # Keep same for simplicity
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, processed_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(processed_channels),
                nn.ReLU(inplace=True)
            ))
            self.bams.append(BAM(processed_channels))
            
        # Final fusion layer after concatenating pyramid outputs
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + len(self.scales) * in_channels, in_channels, 
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        pyramid_features = [x] # Start with the original features
        
        for i, scale in enumerate(self.scales):
            # Apply pooling for different scales
            if scale > 1:
                pooled_x = F.adaptive_avg_pool2d(x, (H // scale, W // scale))
            else:
                pooled_x = x # Scale 1 is the original

            # Process with 1x1 conv and BAM
            processed_x = self.convs[i](pooled_x)
            attended_x = self.bams[i](processed_x)
            
            # Upsample back to original size
            upsampled_x = F.interpolate(attended_x, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled_x)
            
        # Concatenate original features and all pyramid features
        fused_features = torch.cat(pyramid_features, dim=1)
        
        # Apply final fusion convolution
        out = self.fuse_conv(fused_features)
        
        # Add residual connection
        return out + x




class ResNetFPNFeatureExtractor(nn.Module):
    """
    Implements the Feature Extractor from diagram (b).
    Uses a ResNet backbone and FPN-like fusion.
    """
    def __init__(self, backbone_name='resnet18', pretrained=True, 
                 out_channels=256, in_channels=3):
        super().__init__()
        
        # 1. Load ResNet backbone using timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True, # Return features from each stage
            in_chans=in_channels,
            out_indices=(1, 2, 3, 4) # We need stages 1, 2, 3, 4
        )
        
        # Get the channel counts from the backbone stages
        backbone_channels = self.backbone.feature_info.channels()
        # Example for resnet18: [64, 128, 256, 512]

        # 2. Create the 1x1 Convs for channel reduction (lateral layers)
        self.lateral_convs = nn.ModuleList()
        for in_c in backbone_channels:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, kernel_size=1))
            
        # 3. Create the final fusion convolutions (segmentation head)
        # Input channels = num_stages * out_channels
        self.fusion_convs = nn.Sequential(
            nn.Conv2d(len(backbone_channels) * out_channels, out_channels, 
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, # Keep the same channel dim
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # The diagram shows one more 1x1 conv in the head, let's add it
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 1. Get feature maps from ResNet stages
        features = self.backbone(x)
        
        processed_features = []
        target_size = features[0].shape[2:] # Size of the C1 feature map (1/4 res)

        # 2. Apply 1x1 convs and upsample
        for i, feature_map in enumerate(features):
            lateral_feature = self.lateral_convs[i](feature_map)
            
            # Upsample if not the first (highest-res) feature map
            if i > 0:
                lateral_feature = F.interpolate(lateral_feature, size=target_size, 
                                                mode='bilinear', align_corners=False)
            processed_features.append(lateral_feature)
            
        # 3. Concatenate all processed & upsampled features
        fused_features = torch.cat(processed_features, dim=1)
        
        # 4. Apply final fusion convolutions
        output_feature_map = self.fusion_convs(fused_features)
        
        return output_feature_map # Shape: (B, out_channels, H/4, W/4)
