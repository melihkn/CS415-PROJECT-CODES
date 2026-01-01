# models/snunet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class DenseProjector(nn.Module):
    """
    [NEW] Dense Contrastive Learning Projector
    Projects feature maps into a lower-dimensional space and normalizes them
    for contrastive learning.
    """
    def __init__(self, in_channels, out_channels=128):
        super(DenseProjector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # L2 Normalization is critical for contrastive learning
        return F.normalize(x, p=2, dim=1)

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM + Dense-CL Support
    def __init__(self, in_ch=3, out_ch=1, use_dense_cl=False):
        super(SNUNet_ECAM, self).__init__()
        self.use_dense_cl = use_dense_cl
        torch.nn.Module.dump_patches = True
        n1 = 32     # The initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Blocks
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        # Nested Decoder Blocks
        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        # Ensemble Channel Attention Module (ECAM)
        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)

        if self.use_dense_cl:
            self.projector = DenseProjector(in_channels=filters[4], out_channels=128)
        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):
        # --- Encoder Path (Image A) ---
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # [MODIFIED] Uncommented deepest layer for Image A to ensure symmetry for Dense-CL
        x4_0A = self.conv4_0(self.pool(x3_0A))
        
        # --- Encoder Path (Image B) ---
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        # [NEW] Dense Contrastive Projection
        if self.use_dense_cl and self.training:
            proj_A = self.projector(x4_0A)
            proj_B = self.projector(x4_0B)
        else:
            proj_A, proj_B = None, None

        # --- Nested Decoder Path ---
        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        # --- ECAM Fusion ---
        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        if self.use_dense_cl and self.training:
            return out, proj_A, proj_B
        else:
            return (out,)
        


def train_snunet_epoch(model, dataloader, criterion_seg, criterion_dense, optimizer, device, epoch, lambda_dense=0.1):
    """
    Custom training loop for SNUNet that handles:
    1. Segmentation Loss (BCE+Dice)
    2. Dense Contrastive Loss (Dense-CL)
    """
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [SNUNet+DenseCL]")
    
    for batch in pbar:
        # Unpack batch
        img_A = batch['image_A'].to(device)
        img_B = batch['image_B'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (Returns tuple: pred_map, proj_A, proj_B)
        preds, proj_A, proj_B = model(img_A, img_B)
        
        # 1. Segmentation Loss
        loss_seg = criterion_seg(preds, labels)
        
        # 2. Dense Contrastive Loss
        loss_dense = criterion_dense(proj_A, proj_B, labels)
        
        # Total Loss
        loss = loss_seg + (lambda_dense * loss_dense)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'Loss': loss.item(), 'Seg': loss_seg.item(), 'Dense': loss_dense.item()})
        
    epoch_loss = running_loss / len(dataloader)
    
    # Note: For metrics like IoU/F1, we rely on the validation step, 
    # but here we return placeholders or simple calculations if needed.
    # For compatibility with main loop, we return loss and 0.0 for metrics during train step
    return epoch_loss, 0.0, 0.0