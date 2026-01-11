# FILE: src/brain_project/train/visualize_m3_slots.py
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from brain_project.modules.m3_invariance.slots_kmeans import soft_kmeans_slots_barrier
from brain_project.modules.m3_invariance.geodesic_slots import geodesic_em_slots


# ---------------------------
# Utils I/O
# ---------------------------

def load_image(path: str, size: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size))
    x = (np.array(im).astype(np.float32) / 255.0)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return x


def gradient_barrier(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: (B,1,H,W) in [0,1]
    returns barrier in [0,1], same shape
    """
    dy = torch.abs(x01[:, :, 1:, :] - x01[:, :, :-1, :])
    dx = torch.abs(x01[:, :, :, 1:] - x01[:, :, :, :-1])

    dy = F.pad(dy, (0, 0, 1, 0))
    dx = F.pad(dx, (1, 0, 0, 0))

    g = dx + dy
    g = g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return g


# ---------------------------
# M3.5 - Lateral inhibition
# ---------------------------

def _gaussian_kernel2d(ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
    assert ksize % 2 == 1, "ksize must be odd"
    ax = torch.arange(ksize, device=device) - (ksize // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def _blur_per_channel(x: torch.Tensor, ksize: int = 11, sigma: float = 2.5) -> torch.Tensor:
    """
    x: (B,K,H,W)
    depthwise gaussian blur
    """
    B, K, H, W = x.shape
    kernel = _gaussian_kernel2d(ksize, sigma, x.device).view(1, 1, ksize, ksize)
    weight = kernel.repeat(K, 1, 1, 1)  # (K,1,ks,ks)
    return F.conv2d(x, weight, padding=ksize // 2, groups=K)


def lateral_inhibition(
    masks: torch.Tensor,
    iters: int = 12,
    alpha: float = 1.4,
    beta: float = 0.10,
    blur_ksize: int = 11,
    blur_sigma: float = 2.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    masks: (B,K,H,W) non-negative (ideally sum_K ~ 1)
    Returns sharpened masks with competition (M3.5).
    - alpha : strength of cross-inhibition
    - beta  : residual self-preservation (prevents total collapse)
    """
    x = masks.clamp_min(0.0)

    # normalize once to start clean
    x = x / (x.sum(dim=1, keepdim=True) + eps)

    for _ in range(iters):
        # local pooled activity per slot
        x_blur = _blur_per_channel(x, ksize=blur_ksize, sigma=blur_sigma)

        # competitor field = sum of other slots locally
        comp = x_blur.sum(dim=1, keepdim=True) - x_blur  # (B,K,H,W) via broadcasting

        # inhibition + small self-residual
        x = x - alpha * comp + beta * x_blur

        # rectify + renormalize
        x = F.relu(x)
        x = x / (x.sum(dim=1, keepdim=True) + eps)

    return x


def labels_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: (B,K,H,W) -> labels: (B,H,W)
    """
    return masks.argmax(dim=1)


# ---------------------------
# MAIN
# ---------------------------

@torch.no_grad()
def main():
    OUT = Path("./runs/m3_slots")
    OUT.mkdir(parents=True, exist_ok=True)

    path = os.environ.get("M3_IMG", "./data/real_images/test/5805a4ae-64f4-4d98-be10-612e0483f1fe.jpg")
    print("Using image:", path)

    # 0) Image
    x = load_image(path, size=256)

    # 1) MiDaS depth (signal brut)
    print("Loading MiDaS via torch.hub…")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    midas.eval()

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    img = np.array(Image.open(path).convert("RGB")).astype(np.float32)

    t_out = transform(img)
    if isinstance(t_out, dict):
        inp = t_out["image"]
    else:
        inp = t_out

    # sécurité : batch dimension
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)



    depth = midas(inp)  # (B,H',W')
    depth = F.interpolate(depth.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False)  # (B,1,H,W)
    depth = depth / (depth.amax(dim=(2, 3), keepdim=True) + 1e-6)

    # 2) "S2" proxy (en attendant ton vrai M2 complet)
    K_feat = 8
    s2 = depth.repeat(1, K_feat, 1, 1)
    s2 = torch.softmax(s2, dim=1)

    # 3) barrier = |∇depth|
    barrier = gradient_barrier(depth)

    # 4) M3.2 soft-kmeans (optionnel, juste pour visu)
    out_km = soft_kmeans_slots_barrier(
        s2,
        barrier=barrier,
        num_slots=6,
        iters=15,
        tau=0.25,
        #w_barrier=3.0,
        seed=0,
    )
    km_masks = out_km.masks  # (B,S,H,W)

    # 5) M3.4 geodesic EM
    out_geo = geodesic_em_slots(
        s2,
        barrier=barrier,
        num_slots=6,
        em_iters=6,
        #tau=0.20,
        #w_barrier=6.0,
        #w_feat=1.0,
        #dijkstra_down=1,
        #seed_min_sep=14,
        #seed=0,
    )
    geo_masks = out_geo.masks  # (B,S,H,W) expected
    geo_labels = out_geo.labels  # (B,H,W) expected

    # 6) M3.5 lateral inhibition on geo masks
    m35_masks = lateral_inhibition(
        geo_masks,
        iters=14,
        alpha=1.6,
        beta=0.12,
        blur_ksize=11,
        blur_sigma=2.2,
    )
    m35_labels = labels_from_masks(m35_masks)

    # 7) VISU
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    axes[0, 0].imshow(x[0].permute(1, 2, 0))
    axes[0, 0].set_title("Image")

    axes[0, 1].imshow(depth[0, 0].cpu(), cmap="gray")
    axes[0, 1].set_title("MiDaS depth (norm)")

    axes[0, 2].imshow(barrier[0, 0].cpu(), cmap="gray")
    axes[0, 2].set_title("Barrier = |∇depth|")

    axes[0, 3].imshow(km_masks[0, 0].cpu(), cmap="gray")
    axes[0, 3].set_title("M3.2 soft-kmeans slot[0]")

    axes[1, 0].imshow(geo_masks[0, 0].cpu(), cmap="gray")
    axes[1, 0].set_title("M3.4 geodesic slot[0]")

    axes[1, 1].imshow(geo_labels[0].cpu())
    axes[1, 1].set_title("M3.4 labels (argmax)")

    axes[1, 2].imshow(m35_masks[0, 0].cpu(), cmap="gray")
    axes[1, 2].set_title("M3.5 inhibited slot[0]")

    axes[1, 3].imshow(m35_labels[0].cpu())
    axes[1, 3].set_title("M3.5 labels (argmax)")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT / "m3_slots_m35.png"
    plt.savefig(out_path, dpi=170)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

