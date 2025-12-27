import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(DenseContrastiveLoss, self).__init__()
        self.T = temperature

    def forward(self, proj_t1, proj_t2, label):
        """
        proj_t1, proj_t2: (Batch, Channel, H_feat, W_feat) -> Genelde 16x16
        label: (Batch, 1, H_img, W_img) -> Genelde 256x256
        """
        
        # 1. Özellik haritasının (Feature Map) boyutunu al (Örn: 16x16)
        batch_size, channels, height_feat, width_feat = proj_t1.shape
        
        # 2. Etiketi (256x256), özellik haritası boyutuna (16x16) küçült
        # Etiketler 0 veya 1 olduğu için 'nearest' interpolasyon kullanıyoruz (bozulmasın diye)
        label_resized = F.interpolate(label, size=(height_feat, width_feat), mode='nearest')

        # 3. Maskeyi küçültülmüş etiketten oluştur
        mask_no_change = (label_resized == 0).float()
        
        # 4. Benzerlik Matrisi Hesapla
        # (Batch, Channel, H, W) -> Einsum ile kanal boyutunu yok edip benzerlik buluyoruz
        sim_matrix = torch.einsum('bchw,bchw->bhw', [proj_t1, proj_t2])
        sim_matrix = sim_matrix / self.T
        
        # 5. Loss Hesapla
        # mask_no_change ile sadece değişmeyen yerleri dikkate alıyoruz
        # Boyutlar artık 16x16 olduğu için çarpma işlemi hatasız çalışacak
        
        # Sayısal kararlılık için 1e-6 ekliyoruz
        loss = -torch.log(torch.exp(sim_matrix) * mask_no_change + 1e-6)
        
        # Maskenin boş olduğu (her yerin değiştiği) durumları engellemek için:
        num_no_change = mask_no_change.sum()
        if num_no_change > 0:
            loss = loss.sum() / num_no_change
        else:
            loss = loss.mean() # Eğer hiç değişmeyen yer yoksa (nadir) ortalama al
        
        return loss