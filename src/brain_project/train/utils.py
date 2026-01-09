from PIL import Image
import torch
from torchvision import transforms


def load_image_tensor(path, size=256):
    img = Image.open(path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    x = tf(img).unsqueeze(0)  # (1,3,H,W)
    return x

