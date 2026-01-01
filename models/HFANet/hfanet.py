import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
from .components.HighFrequencyExtractor import HighFrequencyExtractor
from .components.DecoderBlock import DecoderBlock

class HFANet(nn.Module):
    def __init__(self, encoder_name='resnet34', classes=1, pretrained='imagenet', in_channels=3, ssl4eo_weights=None, ssl_method='moco'):
        super(HFANet, self).__init__()
        
        # --- 1. Spatial Stream (Backbone) ---
        # We use smp to get a pre-built U-Net with a ResNet encoder.
        if ssl4eo_weights == 'torchgeo':
            # Use TorchGeo's built-in weights
            print("[HFANet] Using TorchGeo SSL4EO-S12 weights")
            self._init_with_torchgeo(encoder_name, in_channels, classes, ssl_method)
            
        elif ssl4eo_weights is not None:
            # Use custom checkpoint file
            print(f"[HFANet] Using SSL4EO-S12 ({ssl_method}) from: {ssl4eo_weights}")
            self.smp_model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=classes
            )
            self._load_ssl4eo_weights(ssl4eo_weights, ssl_method)
            
        else:
            # ImageNet weights
            print(f"[HFANet] Using {encoder_name} with ImageNet weights")
            self.smp_model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=pretrained,
                in_channels=in_channels,
                classes=classes
            )
        
        # --- 2. High-Frequency Stream ---
        self.hf_extractor = HighFrequencyExtractor(in_channels=in_channels)
        
        # The smp ResNet34 encoder's first feature map (sp_features[1])
        # has 64 channels and is at 1/2 resolution.
        # Our HF features (hf_diff) will have (in_channels * 2) channels
        # and be at full resolution.
        hf_channels = in_channels * 2
        
        # Get the channel count of the first spatial feature map
        sp_channels = self.smp_model.encoder.out_channels[1] # e.g., 64 for resnet34

        # --- 3. HFA Module ---
        # This module will create the attention mask.
        # It takes the HF diff map, downsamples it to match the spatial map,
        # and creates a 1-channel attention mask.
        self.hf_attention_head = nn.Sequential(
            nn.Conv2d(hf_channels, hf_channels, kernel_size=3, stride=2, padding=1, bias=False), # Downsample to 1/2 res
            nn.BatchNorm2d(hf_channels),
            nn.ReLU(),
            nn.Conv2d(hf_channels, 1, kernel_size=1, bias=False), # Reduce to 1 channel
            nn.Sigmoid()
        )
        
        # This module fuses the *attended* spatial features
        # It takes the (attended spatial diff) + (raw spatial diff)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sp_channels * 2, sp_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(sp_channels),
            nn.ReLU()
        )

    def _init_with_torchgeo(self, encoder_name, in_channels, classes, ssl_method):
        """Initialize using TorchGeo's pretrained weights."""
        try:
            from torchgeo.models import ResNet50_Weights, resnet50
            
            if ssl_method == 'moco':
                weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
            else:
                # TorchGeo might not have DINO, fall back to MoCo
                weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
                print("[HFANet] DINO not available in TorchGeo, using MoCo")
            
            # Get pretrained backbone
            torchgeo_model = resnet50(weights=weights)
            
            # Create SMP model without weights
            self.smp_model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights=None,
                in_channels=in_channels,
                classes=classes
            )
            
            # Copy weights from TorchGeo to SMP encoder
            smp_state = self.smp_model.encoder.state_dict()
            tg_state = torchgeo_model.state_dict()
            
            loaded = 0
            for key in smp_state:
                if key in tg_state and smp_state[key].shape == tg_state[key].shape:
                    smp_state[key] = tg_state[key]
                    loaded += 1
            
            self.smp_model.encoder.load_state_dict(smp_state)
            print(f"[HFANet] Loaded {loaded} weights from TorchGeo")
            
        except ImportError:
            raise ImportError("TorchGeo not installed. Run 'pip install torchgeo' to use SSL4EO weights.")
        
    def _load_ssl4eo_weights(self, checkpoint_path, ssl_method):
        """Load SSL4EO weights from checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if ssl_method == 'moco':
            state_dict = checkpoint.get('state_dict', checkpoint)
            prefixes = ['module.encoder_q.', 'encoder_q.', 'module.']
        elif ssl_method == 'dino':
            if 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
            else:
                state_dict = checkpoint.get('state_dict', checkpoint)
            prefixes = ['module.', 'backbone.', 'encoder.']
        else:
            raise ValueError(f"Unknown ssl_method: {ssl_method}")
        
        # Process weights
        processed = {}
        skip_keywords = ['fc', 'head', 'projector', 'dino_head', 'mlp', 'pred']
        
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            
            if any(kw in new_key.lower() for kw in skip_keywords):
                continue
            processed[new_key] = value
        
        # Load into encoder
        encoder_state = self.smp_model.encoder.state_dict()
        loaded = 0
        
        for key, value in processed.items():
            if key in encoder_state and encoder_state[key].shape == value.shape:
                encoder_state[key] = value
                loaded += 1
        
        self.smp_model.encoder.load_state_dict(encoder_state)
        print(f"[HFANet] Loaded {loaded}/{len(encoder_state)} weights") 


    def forward(self, x1, x2):
        
        # --- 1. High-Frequency (HF) Stream ---
        hf_t1 = self.hf_extractor(x1) # (B, 6, 256, 256)
        hf_t2 = self.hf_extractor(x2) # (B, 6, 256, 256)
        
        # Get the high-frequency difference
        hf_diff = torch.abs(hf_t1 - hf_t2)

        # --- 2. Spatial (Semantic) Stream ---
        # Get the list of features from the encoder for both images
        sp_features_t1 = self.smp_model.encoder(x1)
        sp_features_t2 = self.smp_model.encoder(x2)

        # --- 3. High-Frequency Attention (HFA) ---
        
        # A) Create the attention mask from the HF stream
        # This mask will be at 1/2 resolution (e.g., 128x128)
        hf_attention_mask = self.hf_attention_head(hf_diff) # (B, 1, 128, 128)
        
        # B) Get the spatial difference at the same 1/2 resolution
        # sp_features[0] is the input, sp_features[1] is the first stage
        sp_diff_s1 = torch.abs(sp_features_t1[1] - sp_features_t2[1]) # (B, 64, 128, 128)
        
        # C) Apply the HFA
        # Multiply the spatial difference by the HF attention mask
        attended_sp_diff_s1 = sp_diff_s1 * hf_attention_mask
        
        # D) Fuse the attended features with the original spatial difference
        # This allows the network to learn how much HF attention to use
        fused_s1 = self.fusion_conv(torch.cat([sp_diff_s1, attended_sp_diff_s1], dim=1))

        # --- 4. Create Fused Feature List for Decoder ---
        # Get the simple differences for all *other* (deeper) stages
        # We must include the difference of the input (index 0) to match the decoder's expectation
        diff_0 = torch.abs(sp_features_t1[0] - sp_features_t2[0])
        
        features_diff = [diff_0, fused_s1] # [Input diff, HFA-fused Stage 1]
        
        for i in range(2, len(sp_features_t1)):
            features_diff.append(torch.abs(sp_features_t1[i] - sp_features_t2[i]))

        # --- 5. Decoder Pass ---
        # Pass the list of *difference features* to the decoder
        change_logits = self.smp_model.decoder(features_diff)
        change_mask = self.smp_model.segmentation_head(change_logits)
        
        return change_mask

class HFANet_timm(nn.Module):
    def __init__(self, encoder_name='resnet34', classes=1, pretrained=True, in_channels=3):
        super(HFANet_timm, self).__init__()
        
        # --- 1. Spatial Stream (Backbone) ---
        # Use timm to get just the backbone
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        
        # Get the channel counts from the backbone
        # For resnet34, this is [64, 64, 128, 256, 512]
        sp_channels = self.backbone.feature_info.channels()

        # --- 2. High-Frequency Stream ---
        self.hf_extractor = HighFrequencyExtractor(in_channels=in_channels)
        hf_channels = in_channels * 2 # 3*2 = 6

        # --- 3. HFA Module (to process the skip connections) ---
        # This module will process the HF diff and the *first* spatial diff map
        # sp_channels[0] is the stem (64 channels, 1/2 res)
        # sp_channels[1] is layer1 (64 channels, 1/4 res)
        
        # Let's apply HFA to the features from stage 1 (1/4 res)
        self.hf_attention_head = nn.Sequential(
            nn.Conv2d(hf_channels, 16, kernel_size=3, stride=2, padding=1, bias=False), # 1/2 res
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1, bias=False), # 1/4 res
            nn.Sigmoid()
        )
        
        # Fusion conv for the HFA-applied features
        # It takes (raw spatial diff) + (attended spatial diff)
        self.fusion_s1 = nn.Sequential(
            nn.Conv2d(sp_channels[1] * 2, sp_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(sp_channels[1]),
            nn.ReLU(inplace=True)
        )

        # --- 4. Manual Decoder ---
        # We now have to build the decoder by hand
        self.dec_layer4 = DecoderBlock(sp_channels[4], sp_channels[3], 256)
        self.dec_layer3 = DecoderBlock(256, sp_channels[2], 128)
        self.dec_layer2 = DecoderBlock(128, sp_channels[1], 64) # HFA will be applied to skip[1]
        self.dec_layer1 = DecoderBlock(64, sp_channels[0], 64)
        
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x1, x2):
        
        # --- 1. High-Frequency Stream ---
        hf_t1 = self.hf_extractor(x1) # (B, 6, 256, 256)
        hf_t2 = self.hf_extractor(x2) # (B, 6, 256, 256)
        hf_diff = torch.abs(hf_t1 - hf_t2)

        # --- 2. Spatial Stream (Siamese) ---
        # Returns a list of 5 feature maps (stem, layer1, layer2, layer3, layer4)
        sp_features_t1 = self.backbone(x1)
        sp_features_t2 = self.backbone(x2)
        
        # Get the differences for all skip connections
        d0 = torch.abs(sp_features_t1[0] - sp_features_t2[0])
        d1 = torch.abs(sp_features_t1[1] - sp_features_t2[1])
        d2 = torch.abs(sp_features_t1[2] - sp_features_t2[2])
        d3 = torch.abs(sp_features_t1[3] - sp_features_t2[3])
        d4 = torch.abs(sp_features_t1[4] - sp_features_t2[4]) # Bottleneck

        # --- 3. Apply HFA Module ---
        # Create mask from HF features (at 1/4 res)
        hf_attention_mask = self.hf_attention_head(hf_diff)
        
        # Apply attention to the corresponding spatial difference map (d1)
        attended_d1 = d1 * hf_attention_mask
        
        # Fuse the attended and original features
        fused_d1 = self.fusion_s1(torch.cat([d1, attended_d1], dim=1))
        
        # --- 4. Manual Decoder Path ---
        # We pass the fused_d1 as the skip connection
        dec4 = self.dec_layer4(d4, d3)
        dec3 = self.dec_layer3(dec4, d2)
        dec2 = self.dec_layer2(dec3, fused_d1) # <- HFA is applied here
        dec1 = self.dec_layer1(dec2, d0)
        
        out = self.final_up(dec1)
        out = self.final_conv(out)
        
        return out
