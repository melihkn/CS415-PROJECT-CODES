import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data.transforms import get_train_transforms, get_val_transforms


class ChangeDetectionDataset(Dataset):
    """
    Universal Change Detection Dataset.
    Compatible with: SNUNet, HDANet, HFANet, STANet, SegNet
    
    Expected directory structure:
        root/
          train/
            A/      (Time 1 images)
            B/      (Time 2 images)
            label/  (Binary change masks)
          val/
            A/
            B/
            label/
          test/
            A/
            B/
            label/
    
    Supported datasets: LEVIR-CD, WHU-CD, S2Looking, DSIFN, etc.
    """
    def __init__(self, root_dir, split='train', img_size=256, transform=None):
        """
        Args:
            root_dir: Root directory of the dataset
            split: One of 'train', 'val', or 'test'
            img_size: Target image size for resizing
            transform: Custom transform pipeline (optional)
        """
        self.root_dir = root_dir
        self.split = split
        
        # Directory paths
        self.dir_A = os.path.join(root_dir, split, 'A')
        self.dir_B = os.path.join(root_dir, split, 'B')
        self.dir_label = os.path.join(root_dir, split, 'label')
        
        # Load and sort filenames
        self.filenames = sorted([
            f for f in os.listdir(self.dir_A) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        
        # Use provided transform or default based on split
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms(img_size) if split == 'train' \
                            else get_val_transforms(img_size)
        
        # Validation check
        if len(self.filenames) == 0:
            raise RuntimeError(f"No images found in '{self.dir_A}'")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # File paths
        path_A = os.path.join(self.dir_A, filename)
        path_B = os.path.join(self.dir_B, filename)
        path_label = os.path.join(self.dir_label, filename)
        
        # Load images
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        label = Image.open(path_label).convert('L')
        
        # Apply transforms
        img_A, img_B, label = self.transform(img_A, img_B, label)
        
        return {
            'image_A': img_A, 
            'image_B': img_B, 
            'label': label, 
            'filename': filename
        }


def get_dataloader(root_dir, batch_size=8, split='train', img_size=256, 
                   num_workers=4, transform=None):
    """
    Creates a DataLoader for change detection.
    Compatible with all models: SNUNet, HDANet, HFANet, STANet, SegNet
    
    Args:
        root_dir: Root directory of the dataset
        batch_size: Batch size for training/evaluation
        split: One of 'train', 'val', or 'test'
        img_size: Target image size
        num_workers: Number of parallel data loading workers
        transform: Custom transform pipeline (optional)
    
    Returns:
        DataLoader instance
    """
    dataset = ChangeDetectionDataset(
        root_dir, 
        split=split, 
        img_size=img_size,
        transform=transform
    )
    
    shuffle = (split == 'train')
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )