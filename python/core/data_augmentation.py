import torchvision.transforms as transforms
import random
import numpy as np
import torch

def get_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.5),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.05)
    ])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

