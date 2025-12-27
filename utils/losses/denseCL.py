import torch
import torch.nn as nn


class DenseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(DenseContrastiveLoss, self).__init__()
        self.T = temperature

    def forward(self, proj_t1, proj_t2, label):
        """
        proj_t1, proj_t2: (Batch, Channel, H, W) - Projector çıktıları
        label: (Batch, 1, H, W) - Ground Truth (0: Değişim Yok, 1: Değişim Var)
        """
        # Değişmeyen alan maskesi (Background/No-change)
        # Label 0 ise T1 ve T2 aynı olmalı -> Pozitif çift
        mask_no_change = (label == 0).float()
        
        # Sadece değişmeyen piksellerdeki benzerliği maksimize etmeye odaklanıyoruz
        # (Basitleştirilmiş Cosine Similarity)
        sim_matrix = torch.einsum('bchw,bchw->bhw', [proj_t1, proj_t2])
        sim_matrix = sim_matrix / self.T
        
        # Pozitif çiftler (Değişim olmayan yerler) için loss
        # Similarity'nin 1'e yakın olmasını istiyoruz (Loss -> 0)
        loss = -torch.log(torch.exp(sim_matrix) * mask_no_change + 1e-6).mean()
        
        return loss