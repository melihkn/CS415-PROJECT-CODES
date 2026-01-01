# models/segnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegNetDecoder(nn.Module):
    """
    Lightweight SegNet-ish decoder:
    - Takes multi-scale encoder features (from SMP encoder)
    - Uses abs(fA - fB) at each scale (siamese)
    - Progressive upsampling + skip fusion
    """
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        """
        encoder_channels: list of channels from SMP encoder .out_channels
            Usually length = 6 for ResNet encoders: [in, c1, c2, c3, c4, c5]
            Features list returned by encoder has same length.
        decoder_channels: 5 stages (from deepest to shallowest fusion)
        """
        super().__init__()
        assert len(decoder_channels) == 5, "decoder_channels must have 5 elements"

        # SMP encoder gives features: [f0, f1, f2, f3, f4, f5] (f5 deepest)
        # We'll decode from f5 -> f1, then output at f0 resolution.
        ch = encoder_channels
        d = decoder_channels

        # stage 5: up f5 and fuse with f4
        self.up5 = ConvBNReLU(ch[5], d[0])
        self.fuse5 = ConvBNReLU(d[0] + ch[4], d[0])

        # stage 4: up and fuse with f3
        self.up4 = ConvBNReLU(d[0], d[1])
        self.fuse4 = ConvBNReLU(d[1] + ch[3], d[1])

        # stage 3: up and fuse with f2
        self.up3 = ConvBNReLU(d[1], d[2])
        self.fuse3 = ConvBNReLU(d[2] + ch[2], d[2])

        # stage 2: up and fuse with f1
        self.up2 = ConvBNReLU(d[2], d[3])
        self.fuse2 = ConvBNReLU(d[3] + ch[1], d[3])

        # stage 1: up to f0 scale (input scale) and refine
        self.up1 = ConvBNReLU(d[3], d[4])
        self.refine = nn.Sequential(
            ConvBNReLU(d[4], d[4]),
            ConvBNReLU(d[4], d[4]),
        )

    def forward(self, diffs):
        """
        diffs: list [df0, df1, df2, df3, df4, df5] where dfi = abs(fAi - fBi)
        """
        df0, df1, df2, df3, df4, df5 = diffs

        # Start from deepest df5
        x = self.up5(df5)
        x = F.interpolate(x, size=df4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse5(torch.cat([x, df4], dim=1))

        x = self.up4(x)
        x = F.interpolate(x, size=df3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, df3], dim=1))

        x = self.up3(x)
        x = F.interpolate(x, size=df2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse3(torch.cat([x, df2], dim=1))

        x = self.up2(x)
        x = F.interpolate(x, size=df1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, df1], dim=1))

        x = self.up1(x)
        x = F.interpolate(x, size=df0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(x)

        return x


class SegNet(nn.Module):
    """
    Siamese (shared encoder) SegNet-style change detector.
    Pretrained encoder is loaded via SMP encoder weights ("imagenet").
    Output: logits [B, 1, H, W]
    """
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()

        weights = "imagenet" if pretrained else None
        self.encoder = smp.encoders.get_encoder(
            name=backbone_name,
            in_channels=in_channels,
            depth=5,
            weights=weights,
        )

        self.decoder = SegNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )

        self.seg_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, x1, x2):
        # encoder returns list of features: [f0..f5]
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        # Siamese difference at each scale
        diffs = [torch.abs(a - b) for a, b in zip(f1, f2)]

        x = self.decoder(diffs)
        logits = self.seg_head(x)
        return logits
