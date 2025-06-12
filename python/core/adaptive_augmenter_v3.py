import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from core.attention_extractor_v3 import extract_attention_map
import torch

class AttentionPreprocessorV3:
    def __init__(self, label=None):
        self.label = label

    def __call__(self, img_pil):
        tensor_img = T.ToTensor()(img_pil)
        attention_map = extract_attention_map(tensor_img, label=self.label)
        tensor_img = tensor_img * torch.tensor(attention_map).unsqueeze(0)
        return tensor_img

class ElasticTransform:
    def __call__(self, img):
        if random.random() < 0.3:
            img = F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(0, 0))
        return img

def get_pipeline(spc, label=None):
    augmentation = T.Compose([
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        ElasticTransform(),
        AttentionPreprocessorV3(label=label)
    ])
    return augmentation

