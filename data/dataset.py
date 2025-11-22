import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class ChangeDetectionDataset(Dataset):
    """
    Template for Change Detection Dataset.
    Assumes data is organized as:
    root/
      A/ (Time 1 images)
      B/ (Time 2 images)
      label/ (Ground truth masks)
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Example directory structure
        self.dir_A = os.path.join(root_dir, split, 'A')
        self.dir_B = os.path.join(root_dir, split, 'B')
        self.dir_label = os.path.join(root_dir, split, 'label')
        
        # Load file list (assuming filenames match across folders)
        # self.filenames = os.listdir(self.dir_A) 
        # Placeholder:
        self.filenames = [] 

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
        label = Image.open(path_label).convert('L') # Grayscale mask
        
        if self.transform:
            # Apply transforms
            # Note: For change detection, geometric transforms must be identical for A, B, and label
            pass
            
        # Convert to tensor
        # img_A = ...
        # img_B = ...
        # label = ...
        
        return {'image_A': img_A, 'image_B': img_B, 'label': label}

def get_dataloader(root_dir, batch_size=8, split='train', num_workers=4):
    dataset = ChangeDetectionDataset(root_dir, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)
