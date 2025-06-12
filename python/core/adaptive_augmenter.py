import torchvision.transforms as transforms
import torch
import random
import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

def get_pipeline(samples_per_class):
    if samples_per_class >= 100:
        strength = 0.1
    elif samples_per_class >= 50:
        strength = 0.2
    elif samples_per_class >= 20:
        strength = 0.3
    elif samples_per_class >= 10:
        strength = 0.4
    else:
        strength = 0.5  # very few-shot

    return transforms.Compose([
        transforms.RandomAffine(
            degrees=int(15 * strength),
            translate=(0.1 * strength, 0.1 * strength),
            scale=(1 - 0.1 * strength, 1 + 0.1 * strength),
            shear=10 * strength
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3 * strength),
        transforms.RandomApply([transforms.ColorJitter(contrast=0.2 * strength)], p=0.5 * strength),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.05 * strength)
    ])

