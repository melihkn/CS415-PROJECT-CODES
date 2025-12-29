import os
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def visualize_results(data_dir, result_dir, num_samples=5, save_path='report_figure.png'):
    """
    Rapor için otomatik karşılaştırma figürü oluşturur.
    Sütunlar: Image A | Image B | Ground Truth | Model Prediction
    """
    

    dir_A = os.path.join(data_dir, 'test', 'A')
    dir_B = os.path.join(data_dir, 'test', 'B')
    dir_label = os.path.join(data_dir, 'test', 'label')
    

    if not os.path.exists(result_dir):
        print(f"HATA: '{result_dir}' klasörü bulunamadı.")
        print("Lütfen önce evaluate.py'yi '--save_predictions' parametresiyle çalıştırın.")
        return

    filenames = [f for f in os.listdir(result_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    if len(filenames) == 0:
        print(f"HATA: '{result_dir}' klasörü boş! Hiç tahmin üretilmemiş.")
        return


    sample_count = min(num_samples, len(filenames))
    selected_files = random.sample(filenames, sample_count)
    
    # Plot ayarları
    fig, axes = plt.subplots(sample_count, 4, figsize=(16, 4 * sample_count))
    
    # Başlıklar
    cols = ['Time 1 (Image A)', 'Time 2 (Image B)', 'Ground Truth', 'SNUNet Prediction']
    
    # Eğer sadece 1 satır varsa axes boyutunu düzelt
    if sample_count == 1:
        axes = axes.reshape(1, -1)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold', pad=20)

    print(f"{sample_count} adet rastgele örnek görselleştiriliyor...")

    for i, filename in enumerate(selected_files):
        # Dosya yolları
        path_A = os.path.join(dir_A, filename)
        path_B = os.path.join(dir_B, filename)
        path_label = os.path.join(dir_label, filename)
        path_pred = os.path.join(result_dir, filename)
        
        # Resimleri Yükle
        try:
            img_A = Image.open(path_A).convert('RGB')
            img_B = Image.open(path_B).convert('RGB')
            label = Image.open(path_label).convert('L')
            pred = Image.open(path_pred).convert('L')
        except FileNotFoundError as e:
            print(f"Dosya bulunamadı hatası: {e}")
            continue
        
        # 1. Sütun: Image A
        axes[i, 0].imshow(img_A)
        axes[i, 0].axis('off')
        
        # 2. Sütun: Image B
        axes[i, 1].imshow(img_B)
        axes[i, 1].axis('off')
        
        # 3. Sütun: Label (Gerçek)
        axes[i, 2].imshow(label, cmap='gray')
        axes[i, 2].axis('off')
        
        # 4. Sütun: Prediction (Tahmin)
        axes[i, 3].imshow(pred, cmap='gray')
        axes[i, 3].axis('off')

    # Kaydet
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n--> ✅ Figür başarıyla kaydedildi: {save_path}")
    print("Dosyalar menüsünden bu resmi indirip raporuna ekleyebilirsin.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Change Detection Results')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset root directory')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing prediction masks')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--save_path', type=str, default='final_report_visuals.png', help='Output filename')
    
    args = parser.parse_args()
    
    visualize_results(args.data_dir, args.result_dir, args.num_samples, args.save_path)
