# Change Detection Project (CS415)

This repository contains PyTorch implementations of state-of-the-art deep learning models for **Change Detection**. The goal of this project is to detect and segment changes between two co-registered images (Time 1 and Time 2) of the same scene.

## 🏗 Implemented Models

The project features modular implementations of the following architectures:

*   **HDANet (Hierarchical Difference Attention Network)**:
    *   Utilizes a **Hierarchical Difference Attention Module** to focus on relevant changes while suppressing noise.
    *   Features an **HRNet** backbone for multi-scale feature extraction.
    *   Includes **ASPP** (Atrous Spatial Pyramid Pooling) for capturing multi-scale context.

*   **HFANet (High-Frequency Attention Network)**:
    *   Incorporates a dedicated **High-Frequency Stream** (using Sobel filters) to capture edge details and fine boundaries.
    *   Uses a **High-Frequency Attention (HFA)** module to refine spatial features with edge information.
    *   Available with both `segmentation-models-pytorch` and `timm` backbones.

*   **STANet (Spatial-Temporal Attention Network)**:
    *   Based on a Siamese ResNet backbone with FPN (Feature Pyramid Network).
    *   Integrates a **Pyramid Attention Module (PAM)** to capture long-range spatial-temporal dependencies between the two time steps.

## 📂 Project Structure

The codebase is organized for modularity and extensibility:

```text
CS415-PROJECT-CODES/
├── data/
│   └── dataset.py          # Dataset loading logic (Custom ChangeDetectionDataset)
├── models/
│   ├── layers.py           # Shared building blocks (ASPP, BAM, PAM, HighFreqExtractor)
│   ├── hdanet.py           # HDANet implementation
│   ├── hfanet.py           # HFANet implementation
│   └── stanet.py           # STANet implementation
├── notebooks/              # Original research and experiment notebooks
├── utils/                  # Utility functions
├── train.py                # Main training script
└── requirements.txt        # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Dataset Preparation

The data loader expects the following directory structure:

```text
dataset_root/
├── train/
│   ├── A/          # Images from Time 1
│   ├── B/          # Images from Time 2
│   └── label/      # Binary Change Masks
├── val/
│   ├── A/
│   ├── B/
│   └── label/
```

### Training

You can train models using the `train.py` script. It supports command-line arguments for easy configuration.

**Basic Usage:**

```bash
python train.py --model hfanet --data_dir /path/to/dataset
```

**Advanced Usage:**

```bash
python train.py \
  --model stanet \
  --backbone resnet34 \
  --data_dir ./dataset \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001
```

**Available Arguments:**

*   `--model`: Choose the model architecture (`hfanet`, `hfanet_timm`, `hdanet`, `stanet`).
*   `--backbone`: Encoder backbone (e.g., `resnet18`, `resnet34`, `hrnet_w18`).
*   `--data_dir`: Path to your dataset root directory.
*   `--epochs`: Number of training epochs (default: 50).
*   `--batch_size`: Batch size (default: 8).
*   `--lr`: Learning rate (default: 1e-4).

## 🛠 Dependencies

*   [PyTorch](https://pytorch.org/)
*   [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
*   [timm](https://github.com/rwightman/pytorch-image-models)
*   NumPy
*   Pillow
