import cv2
import numpy as np

def extract_attention_map(image_tensor):
    # Tensor to numpy conversion (expecting shape: (1,28,28))
    image_np = image_tensor.squeeze().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Sobel gradients
    grad_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize attention map
    magnitude = magnitude / (np.max(magnitude) + 1e-8)
    return magnitude.astype(np.float32)

