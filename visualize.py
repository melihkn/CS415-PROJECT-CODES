import os
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def visualize_results(data_dir, result_dir, num_samples=5, save_path='report_figure.png'):
    # Klasör Yolları
    dir_A = os.path.join(data_dir, 'test', 'A')
    dir_B = os.path.join(data_dir, 'test', 'B')
    dir_label = os.path.join(data_dir, 'test', 'label')
    
    # Prediction klasöründeki dosyaları al
    if not os.path.exists(result_dir):
        print(f"Hata: Klasör bulunamadı: {result_dir}")
        return

    pred_files = [f for f in os.listdir(result_dir) if f.endswith(('.png', '.jpg'))]
    
    if not pred_files:
        print("Tahmin klasörü boş!")
        return

    # Rastgele seçim yap
    selected_files = random.sample(pred_files, min(num_samples, len(pred_files)))
    
    # Plot Ayarları
    fig, axes = plt.subplots(len(selected_files), 4, figsize=(16, 4 * len(selected_files)))
    cols = ['Time 1 (Image A)', 'Time 2 (Image B)', 'Ground Truth', 'SNUNet Prediction']
    
    if len(selected_files) == 1: axes = axes.reshape(1, -1)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=10)

    print(f"{len(selected_files)} örnek görselleştiriliyor...")

    for i, filename in enumerate(selected_files):
        # Dosya isimleri ARTIK AYNI olduğu için doğrudan isimle çağırıyoruz
        path_A = os.path.join(dir_A, filename)
        path_B = os.path.join(dir_B, filename)
        path_label = os.path.join(dir_label, filename)
        path_pred = os.path.join(result_dir, filename)
        
        try:
            img_A = Image.open(path_A).convert('RGB')
            img_B = Image.open(path_B).convert('RGB')
            label = Image.open(path_label).convert('L')
            pred = Image.open(path_pred).convert('L')

            axes[i, 0].imshow(img_A); axes[i, 0].axis('off')
            axes[i, 1].imshow(img_B); axes[i, 1].axis('off')
            axes[i, 2].imshow(label, cmap='gray'); axes[i, 2].axis('off')
            axes[i, 3].imshow(pred, cmap='gray'); axes[i, 3].axis('off')
            
            # Dosya ismini kenara yazalım (Referans için)
            axes[i, 0].text(0, -10, filename, fontsize=10, color='blue')

        except FileNotFoundError:
            print(f"UYARI: {filename} dosyası kaynak klasörlerde bulunamadı!")
            continue

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"--> Figür kaydedildi: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='final_visuals.png')
    args = parser.parse_args()
    
    visualize_results(args.data_dir, args.result_dir, args.num_samples, args.save_path)