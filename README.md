# Change Detection Project (CS415)

This repository contains PyTorch implementations of state-of-the-art deep learning models for **Building Change Detection** from bi-temporal satellite/aerial images. The goal is to detect and segment changes between two co-registered images (Time 1 and Time 2) of the same scene.

## 🏗 Implemented Models

The project features modular implementations of the following architectures:

| Model | Description | Backbone |
|-------|-------------|----------|
| **SNUNet-CD** | Densely connected Siamese network with Ensemble Channel Attention Module (ECAM) for preserving fine-grained localization | Custom |
| **HDANet** | Hierarchical Difference Attention Network with ASPP for multi-scale context | HRNet |
| **HFANet** | High-Frequency Attention Network using Sobel filters for edge-aware detection | ResNet (smp) |
| **HFANet-TIMM** | HFANet variant with timm backbone support | ResNet (timm) |
| **STANet** | Spatial-Temporal Attention Network with Pyramid Attention Module (PAM) | ResNet + FPN |

## 📂 Project Structure

```text
CS415-PROJECT-CODES/
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Universal dataset loader for all models
│   └── transforms.py           # Synchronized augmentations for image pairs
│
├── models/
│   ├── HDANet/                 # HDANet model and components
│   ├── HFANet/                 # HFANet model and components
│   ├── __init__.py
│   ├── layers.py               # Shared blocks (ASPP, BAM, PAM, etc.)
│   ├── snunet.py               # SNUNet-CD implementation
│   └── stanet.py               # STANet implementation
│
├── utils/
│   ├── __init__.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dice.py             # Dice Loss
│   │   ├── focal.py            # Focal Loss
│   │   └── soft_iou.py         # Soft IoU Loss
│   ├── metrics.py              # IoU, F1, Precision, Recall, etc.
│   ├── training.py             # Training utilities
│   └── evaluation.py           # Validation and testing utilities
│
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── colab_runner.ipynb
│   ├── hdanet.ipynb
│   ├── hfanet.ipynb
│   ├── hfa_timm.ipynb
│   └── stanet.ipynb
│
├── train.py                    # Main training script (CLI)
├── test.py                     # Main testing script (CLI)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset Preparation

The data loader expects the following directory structure:

```text
dataset_root/
├── train/
│   ├── A/              # Time 1 images
│   ├── B/              # Time 2 images
│   └── label/          # Binary change masks (0: no change, 255: change)
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

**Supported Datasets:** LEVIR-CD, WHU-CD, S2Looking, DSIFN, etc.

## 🎯 Training

### Basic Usage

```bash
python train.py --model snunet --data_dir /path/to/dataset
```

### Advanced Usage

```bash
python train.py \
    --model hfanet \
    --backbone resnet50 \
    --data_dir ./dataset \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --loss bce+dice \
    --optimizer adamw \
    --scheduler cosine
```

### Training Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Model architecture | `snunet` | `snunet`, `hdanet`, `hfanet`, `hfanet_timm`, `stanet` |
| `--backbone` | Encoder backbone | `resnet34` | Any supported by smp/timm |
| `--data_dir` | Dataset root directory | Required | - |
| `--img_size` | Input image size | `256` | - |
| `--epochs` | Number of epochs | `100` | - |
| `--batch_size` | Batch size | `8` | - |
| `--lr` | Learning rate | `1e-4` | - |
| `--loss` | Loss function | `bce+dice` | `bce`, `dice`, `focal`, `softiou`, `bce+dice`, `bce+focal` |
| `--optimizer` | Optimizer | `adamw` | `adamw`, `adam`, `sgd` |
| `--scheduler` | LR scheduler | `plateau` | `plateau`, `cosine`, `step`, `none` |
| `--patience` | Early stopping patience | `15` | - |
| `--checkpoint_dir` | Checkpoint save directory | `./checkpoints` | - |

## 🧪 Testing

### Basic Usage

```bash
python test.py \
    --model snunet \
    --checkpoint ./checkpoints/best_snunet.pth \
    --data_dir ./dataset
```

### Full Evaluation with Outputs

```bash
python test.py \
    --model hfanet \
    --backbone resnet50 \
    --checkpoint ./checkpoints/best_hfanet.pth \
    --data_dir ./dataset \
    --output_dir ./results/hfanet \
    --save_predictions \
    --save_pr_curve
```

### Testing Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture | Required |
| `--checkpoint` | Path to model checkpoint | Required |
| `--data_dir` | Dataset root directory | Required |
| `--output_dir` | Results output directory | `./results` |
| `--save_predictions` | Save prediction images | `False` |
| `--save_pr_curve` | Save PR curve plot | `False` |
| `--save_metrics` | Save metrics to JSON | `True` |

## 📓 Notebook Usage

For interactive experimentation in Jupyter notebooks:

```python
import sys
sys.path.append('..')  # Add root to path

import torch
from data.dataset import get_dataloader
from models.snunet import SNUNet_ECAM
from utils.losses import DiceLoss
from utils.training import train_one_epoch, CombinedLoss
from utils.evaluation import validate, evaluate_on_loader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SNUNet_ECAM(in_ch=3, out_ch=1).to(device)

# Data
train_loader = get_dataloader('../dataset', split='train', batch_size=8)
val_loader = get_dataloader('../dataset', split='val', batch_size=8)

# Loss
criterion = CombinedLoss([
    (torch.nn.BCEWithLogitsLoss(), 1.0),
    (DiceLoss(), 1.0),
])

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Train one epoch
loss, iou, f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1)

# Validate
val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
```

## 📊 Metrics

The following metrics are computed during training and evaluation:

- **IoU (Intersection over Union)** - Primary metric for segmentation quality
- **F1-Score** - Harmonic mean of precision and recall
- **Precision** - Ratio of true positives to predicted positives
- **Recall** - Ratio of true positives to actual positives
- **AP (Average Precision)** - Area under the PR curve

## 🛠 Dependencies

- [PyTorch](https://pytorch.org/) >= 1.10
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [scikit-learn](https://scikit-learn.org/)
- NumPy
- Pillow
- tqdm
- matplotlib

## 📚 References

- **SNUNet-CD**: Fang et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images"
- **HDANet**: Wang et al., "A High-Resolution Feature Difference Attention Network for Building Change Detection"
- **HFANet**: Zheng et al., "HFA-Net: High Frequency Attention Siamese Network for Building Change Detection"
- **STANet**: Chen & Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"

## 👥 Team

**Group 9 - Sabancı University CS415**

- Yiğit Demirkan
- Buğrahan Yapılmışev
- Melih Kaan Şahinbaş
- Ahmet Faruk Keskin
- Ahmet Çalışkan

## 📄 License

This project is for educational purposes as part of CS415 - Introduction to Deep Learning course.