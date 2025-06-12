import cv2
import numpy as np

def extract_attention_map(image_tensor):
    # Tensor to numpy (expecting shape: (1,28,28))
    image_np = image_tensor.squeeze().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # ORB Keypoint extraction
    orb = cv2.ORB_create(nfeatures=50)
    keypoints = orb.detect(image_np, None)

    # Create blank attention map
    attention_map = np.zeros_like(image_np, dtype=np.float32)

    # Mark keypoints on attention map
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < 28 and 0 <= y < 28:
            attention_map[y, x] = 1.0

    # Slight Gaussian blur for smoothness
    attention_map = cv2.GaussianBlur(attention_map, (3, 3), sigmaX=0.5)

    # Normalize attention
    attention_map /= (attention_map.max() + 1e-8)

    # Soft attention: do not zero the rest, only modulate
    soft_attention = 0.5 + 0.5 * attention_map  # Soft scaling: between 0.5 and 1.0

    return soft_attention.astype(np.float32)

