import cv2
import numpy as np

# Nombre de keypoints adaptatif selon la classe (prototype simplifié)
CLASS_KEYPOINTS = {
    0: 20, 1: 10, 2: 20, 3: 30, 4: 15,
    5: 25, 6: 30, 7: 10, 8: 40, 9: 35
}

def extract_attention_map(image_tensor, label=None):
    image_np = image_tensor.squeeze().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    nfeatures = CLASS_KEYPOINTS.get(label, 20)  # Default = 20 keypoints
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints = orb.detect(image_np, None)

    attention_map = np.zeros_like(image_np, dtype=np.float32)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < 28 and 0 <= y < 28:
            attention_map[y, x] = 1.0

    attention_map = cv2.GaussianBlur(attention_map, (3, 3), sigmaX=0.5)
    attention_map /= (attention_map.max() + 1e-8)
    soft_attention = 0.5 + 0.5 * attention_map
    return soft_attention.astype(np.float32)

