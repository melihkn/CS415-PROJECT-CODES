import random
from PIL import Image
import torchvision.transforms.functional as TF


class Compose:
    """Applies a sequence of transforms to image pairs and label."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_A, img_B, label):
        for t in self.transforms:
            img_A, img_B, label = t(img_A, img_B, label)
        return img_A, img_B, label


class Resize:
    """Resizes all three images to the specified size."""
    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else [size, size]
    
    def __call__(self, img_A, img_B, label):
        img_A = TF.resize(img_A, self.size, interpolation=Image.BILINEAR)
        img_B = TF.resize(img_B, self.size, interpolation=Image.BILINEAR)
        label = TF.resize(label, self.size, interpolation=Image.NEAREST)
        return img_A, img_B, label


class RandomHorizontalFlip:
    """Randomly flips images horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            return TF.hflip(img_A), TF.hflip(img_B), TF.hflip(label)
        return img_A, img_B, label


class RandomVerticalFlip:
    """Randomly flips images vertically with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            return TF.vflip(img_A), TF.vflip(img_B), TF.vflip(label)
        return img_A, img_B, label


class RandomRotation:
    """Randomly rotates images by 90, 180, or 270 degrees."""
    def __init__(self, p=0.5, angles=[90, 180, 270]):
        self.p = p
        self.angles = angles

    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return TF.rotate(img_A, angle), TF.rotate(img_B, angle), TF.rotate(label, angle)
        return img_A, img_B, label


class ColorJitter:
    """
    Applies synchronized brightness and contrast augmentation to both images.
    Only applied to img_A and img_B, not to label.
    Same values are applied to both images to maintain consistency.
    """
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p
    
    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            
            img_A = TF.adjust_brightness(img_A, brightness_factor)
            img_A = TF.adjust_contrast(img_A, contrast_factor)
            img_B = TF.adjust_brightness(img_B, brightness_factor)
            img_B = TF.adjust_contrast(img_B, contrast_factor)
        return img_A, img_B, label


class GaussianBlur:
    """Applies Gaussian blur augmentation to both images."""
    def __init__(self, kernel_size=3, p=0.3):
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            img_A = TF.gaussian_blur(img_A, self.kernel_size)
            img_B = TF.gaussian_blur(img_B, self.kernel_size)
        return img_A, img_B, label


class ToTensor:
    """Converts PIL Images to PyTorch Tensors."""
    def __call__(self, img_A, img_B, label):
        return TF.to_tensor(img_A), TF.to_tensor(img_B), TF.to_tensor(label)


class Normalize:
    """
    Normalizes images using ImageNet statistics.
    Only applied to img_A and img_B, not to label.
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, img_A, img_B, label):
        img_A = TF.normalize(img_A, self.mean, self.std)
        img_B = TF.normalize(img_B, self.mean, self.std)
        return img_A, img_B, label


class BinarizeLabel:
    """
    Converts label to binary (0/1).
    Cleans up interpolation artifacts that may occur after resizing.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, img_A, img_B, label):
        label = (label > self.threshold).float()
        return img_A, img_B, label


# ============================================================
# Helper Functions - Ready-to-use transform pipelines
# Compatible with: SNUNet, HDANet, HFANet, STANet, SegNet
# ============================================================

def get_train_transforms(img_size=256):
    """
    Returns training transform pipeline with augmentations.
    
    Args:
        img_size: Target image size (int or tuple)
    
    Returns:
        Compose object with training transforms
    """
    return Compose([
        Resize(img_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        GaussianBlur(kernel_size=3, p=0.2),
        ToTensor(),
        Normalize(),
        BinarizeLabel(),
    ])


def get_val_transforms(img_size=256):
    """
    Returns validation/test transform pipeline without augmentations.
    
    Args:
        img_size: Target image size (int or tuple)
    
    Returns:
        Compose object with validation transforms
    """
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(),
        BinarizeLabel(),
    ])