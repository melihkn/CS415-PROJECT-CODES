import random
import torchvision.transforms.functional as TF

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_A, img_B, label):
        for t in self.transforms:
            img_A, img_B, label = t(img_A, img_B, label)
        return img_A, img_B, label

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            return TF.hflip(img_A), TF.hflip(img_B), TF.hflip(label)
        return img_A, img_B, label

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            return TF.vflip(img_A), TF.vflip(img_B), TF.vflip(label)
        return img_A, img_B, label

class RandomRotation(object):
    def __init__(self, p=0.5, angles=[90, 180, 270]):
        self.p = p
        self.angles = angles

    def __call__(self, img_A, img_B, label):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return TF.rotate(img_A, angle), TF.rotate(img_B, angle), TF.rotate(label, angle)
        return img_A, img_B, label

class ToTensor(object):
    def __call__(self, img_A, img_B, label):
        return TF.to_tensor(img_A), TF.to_tensor(img_B), TF.to_tensor(label)
