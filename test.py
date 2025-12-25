import os
import torch
from torch.utils.data import DataLoader
from models.snunet import SNUNet_ECAM
from data.dataset import ChangeDetectionDataset
import torchvision.transforms.functional as TF
from PIL import Image

def test(data_dir, model_path, output_dir='./results'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Modeli Yükle
    print("Model yükleniyor...")
    model = SNUNet_ECAM(in_ch=3, out_ch=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Test Verisini Yükle
    test_dataset = ChangeDetectionDataset(root_dir=data_dir, split='test', img_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Tek tek işleyelim

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Test başlıyor. Toplam {len(test_dataset)} görüntü...")

    # 3. Tahmin Döngüsü
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            filename = batch['filename'][0] # Dosya ismini al

            # Model Tahmini
            outputs = model(img_A, img_B)
            
            # SNUNet tuple döndürür, en son (en iyi) çıktıyı alıyoruz: outputs[0]
            # Sigmoid ile 0-1 arasına sıkıştırıyoruz
            pred = torch.sigmoid(outputs[0])
            
            # 0.5 Threshold uygulayıp Binary Maske yapıyoruz
            pred_mask = (pred > 0.5).float().cpu().squeeze() # Boyutları (H, W) yap
            
            # Resmi Kaydet
            pred_pil = TF.to_pil_image(pred_mask)
            save_path = os.path.join(output_dir, filename)
            pred_pil.save(save_path)
            
            if i % 10 == 0:
                print(f"İşlenen: {i}/{len(test_dataset)}")

    print(f"Test tamamlandı! Sonuçlar '{output_dir}' klasörüne kaydedildi.")

if __name__ == '__main__':
    # Kullanım:
    # 1. Önce 'train.py' çalıştırıp 'checkpoints/best_model.pth' oluşturmalısın.
    # 2. Sonra bu scripti çalıştır.
    test(data_dir='./dataset', model_path='checkpoints/best_model.pth')
