# FILE: src/brain_project/train/visualize_m3_slots.py
from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from brain_project.modules.m3_invariance.slots_kmeans import soft_kmeans_slots_barrier
from brain_project.modules.m3_invariance.geodesic_slots import geodesic_em_slots


def _find_default_image() -> str:
    for d in ["./data/real_images/test", "./data/real_images"]:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            imgs = sorted(glob.glob(os.path.join(d, ext)))
            if imgs:
                return imgs[0]
    raise SystemExit("No images found in ./data/real_images or ./data/real_images/test")


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(im: Image.Image) -> torch.Tensor:
    """(1,3,H,W) float32 in [0,1]"""
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


def preprocess_for_midas(im: Image.Image, size: int = 256) -> torch.Tensor:
    """
    MiDaS_small accepte un tensor float (B,3,H,W).
    On fait un preprocess stable (sans dépendre des transforms du hub).
    """
    im = im.resize((size, size), resample=Image.BILINEAR)
    x = pil_to_tensor(im)  # [0,1]
    # Normalisation type ImageNet (suffisant et stable)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def gradient_barrier(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,1,H,W) -> barrier: (B,1,H,W) in [0,1]
    Gradient magnitude (L1) with padding to keep same size.
    """
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = F.pad(dy, (0, 0, 1, 0))
    dx = F.pad(dx, (1, 0, 0, 0))
    g = dx + dy
    g = g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return g


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@torch.no_grad()
def main():
    path = os.environ.get("M3_IMG", _find_default_image())
    out_dir = os.environ.get("M3_OUT_DIR", "./runs/m3_slots")
    img_size = int(os.environ.get("M3_IMG_SIZE", "256"))

    k_regions = int(os.environ.get("M3_K_REGIONS", "8"))
    k_slots = int(os.environ.get("M3_K_SLOTS", "6"))
    km_iters = int(os.environ.get("M3_KM_ITERS", "10"))
    geo_iters = int(os.environ.get("M3_GEO_ITERS", "6"))

    os.makedirs(out_dir, exist_ok=True)

    print("Using image:", path)

    pil = load_pil(path)
    pil_resized = pil.resize((img_size, img_size), resample=Image.BILINEAR)
    x_vis = pil_to_tensor(pil_resized)  # for display

    # ---- MiDaS depth (robuste, sans transforms hub) ----
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    midas.eval()

    inp = preprocess_for_midas(pil, size=img_size)  # (1,3,H,W) normalized
    depth = midas(inp)  # (1,H',W') or (1,H,W) depending model
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)  # (1,1,H,W)
    elif depth.ndim == 4 and depth.shape[1] != 1:
        # rare, but just in case
        depth = depth.mean(dim=1, keepdim=True)

    depth = F.interpolate(depth, size=(img_size, img_size), mode="bilinear", align_corners=False)
    depth = depth - depth.amin(dim=(2, 3), keepdim=True)
    depth = depth / (depth.amax(dim=(2, 3), keepdim=True) + 1e-6)

    # ---- "S2" perceptif minimal : K canaux = projection soft de la depth ----
    # Ici on reste fidèle à ton objectif: différenciation grossière de zones.
    s2 = depth.repeat(1, k_regions, 1, 1)
    s2 = torch.softmax(s2, dim=1)

    # ---- Barrière = gradient (contours) ----
    barrier = gradient_barrier(depth)

    # ---- M3.2 : soft-kmeans avec barrière ----
    out_km = soft_kmeans_slots_barrier(
        s2, barrier=barrier, num_slots=k_slots, iters=km_iters
    )

    # ---- M3.4 : slots géodésiques ----
    out_geo = geodesic_em_slots(
        w_feat=s2,
        barrier=barrier,
        num_slots=k_slots,
        em_iters=geo_iters
    )


    # ---- Visualize ----
    img = to_np(x_vis[0].permute(1, 2, 0))
    dep = to_np(depth[0, 0])
    bar = to_np(barrier[0, 0])

    # Les sorties peuvent être dataclasses selon ton implémentation
    km_masks = out_km.masks if hasattr(out_km, "masks") else out_km
    geo_masks = out_geo.masks if hasattr(out_geo, "masks") else out_geo
    geo_labels = out_geo.labels if hasattr(out_geo, "labels") else None

    km0 = to_np(km_masks[0, 0])
    geo0 = to_np(geo_masks[0, 0])

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(dep, cmap="gray")
    plt.title("MiDaS depth (normalized)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(bar, cmap="gray")
    plt.title("Barrier = |∇depth|")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(km0, cmap="gray")
    plt.title("M3.2 soft-kmeans slot[0]")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(geo0, cmap="gray")
    plt.title("M3.4 geodesic slot[0]")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    if geo_labels is not None:
        plt.imshow(to_np(geo_labels[0]), cmap="tab20")
        plt.title("M3.4 labels (argmax)")
    else:
        plt.imshow(np.zeros((img_size, img_size)), cmap="gray")
        plt.title("M3.4 labels: n/a")
    plt.axis("off")

    out_path = os.path.join(out_dir, "m3_slots.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

