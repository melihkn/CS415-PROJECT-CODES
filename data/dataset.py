import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ChangeDetectionDataset(Dataset):
    """
    Custom Dataset for Change Detection (SNUNet-CD uyumlu).
    Structure:
    root/
      train/
        A/ (Time 1 images)
        B/ (Time 2 images)
        label/ (Binary masks)
    """
    def __init__(self, root_dir, split='train', img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Klasör yolları
        self.dir_A = os.path.join(root_dir, split, 'A')
        self.dir_B = os.path.join(root_dir, split, 'B')
        self.dir_label = os.path.join(root_dir, split, 'label')
        
        # Dosya isimlerini yükle ve sırala (A ve B'nin karışmaması için sorted şart)
        # Sadece resim dosyalarını (.png, .jpg) alıyoruz
        self.filenames = sorted([
            f for f in os.listdir(self.dir_A) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        
        # Basit bir kontrol
        if len(self.filenames) == 0:
            print(f"UYARI: '{self.dir_A}' içinde hiç dosya bulunamadı!")

    def transform(self, img_A, img_B, label):
        """
        A, B ve Label için SENKRONİZE dönüşümler.
        """
        # 1. Resize
        # Label için Interpolation.NEAREST kullanıyoruz ki kenarlar bozulmasın (0 veya 1 kalsın)
        img_A = TF.resize(img_A, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        img_B = TF.resize(img_B, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        label = TF.resize(label, [self.img_size, self.img_size], interpolation=Image.NEAREST)

        # 2. Augmentation (Sadece eğitim setinde)
        if self.split == 'train':
            # Horizontal Flip
            if random.random() > 0.5:
                img_A = TF.hflip(img_A)
                img_B = TF.hflip(img_B)
                label = TF.hflip(label)

            # Vertical Flip
            if random.random() > 0.5:
                img_A = TF.vflip(img_A)
                img_B = TF.vflip(img_B)
                label = TF.vflip(label)

            # Random Rotation (90 derece katları)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img_A = TF.rotate(img_A, angle)
                img_B = TF.rotate(img_B, angle)
                label = TF.rotate(label, angle)

        # 3. To Tensor & Normalize
        # Label'ı tensor'a çevir (Normalization yok!)
        t_label = TF.to_tensor(label)
        
        # Resimler için ImageNet istatistikleri (Backbone modelleri için standarttır)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        t_img_A = TF.to_tensor(img_A)
        t_img_A = TF.normalize(t_img_A, mean=mean, std=std)
        
        t_img_B = TF.to_tensor(img_B)
        t_img_B = TF.normalize(t_img_B, mean=mean, std=std)

        return t_img_A, t_img_B, t_label

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        path_A = os.path.join(self.dir_A, filename)
        path_B = os.path.join(self.dir_B, filename)
        path_label = os.path.join(self.dir_label, filename)
        
        # Load images
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        
        # Label okuma (Eğer maske dosyası yoksa hata vermesin diye try-except koyabilirsin ama şimdilik gerek yok)
        # Label 'L' (Grayscale) modunda açılır. 0=Siyah, 255=Beyaz
        label = Image.open(path_label).convert('L') 
        
        # Dönüşümleri uygula
        img_A, img_B, label = self.transform(img_A, img_B, label)
        
        # Binary Maske Garantisi: Piksel değerlerini 0 ve 1'e zorla
        # (Resize sırasında kenarlar grileşmiş olabilir, onları temizleriz)
        label = (label > 0.5).float()
        
        return {'image_A': img_A, 'image_B': img_B, 'label': label, 'filename': filename}

def get_dataloader(root_dir, batch_size=8, split='train', img_size=256, num_workers=4):
    """
    Dataloader oluşturucu fonksiyon.
    """
    dataset = ChangeDetectionDataset(root_dir, split=split, img_size=img_size)
    
    # Validation/Test için shuffle kapalı olmalı
    shuffle = True if split == 'train' else False
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True, # GPU kullanıyorsan performansı artırır
        drop_last=(split == 'train') # Batch size tam bölünmezse sondaki eksik batch'i atar (stabilite için)
    )