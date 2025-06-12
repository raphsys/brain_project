import torchvision.transforms as T
from core.attention_extractor_v2 import extract_attention_map
import torch

class AttentionPreprocessor:
    def __init__(self):
        pass

    def __call__(self, img_pil):
        tensor_img = T.ToTensor()(img_pil)
        attention_map = extract_attention_map(tensor_img)
        tensor_img = tensor_img * torch.tensor(attention_map).unsqueeze(0)
        return tensor_img

def get_pipeline(spc, class_variance=None):
    jitter = 0.2
    rotation = 15
    translate = (0.1, 0.1)

    augmentation = T.Compose([
        T.RandomAffine(degrees=rotation, translate=translate),
        AttentionPreprocessor()  # Soft attention appliqué proprement ici
    ])
    return augmentation

