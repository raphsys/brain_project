# ============================================================
# visualize_m3_slots.py
# Visualisation propre du pipeline M3 (+ M4)
# MiDaS DPT_Large OBLIGATOIRE
# ============================================================

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from brain_project.modules.m3_invariance.m3_pipeline import run_m3


# ============================================================
# Utils
# ============================================================

def load_rgb(path: str, size: int = 256) -> torch.Tensor:
    """
    Load RGB image → (1,3,H,W) float32 in [0,1]
    """
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def load_midas(device: torch.device):
    """
    Load MiDaS DPT_Large + dpt_transform (OBLIGATOIRE).
    """
    print("Loading MiDaS (DPT_Large)…")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
    midas = midas.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = transforms.dpt_transform

    return midas, midas_transform


@torch.no_grad()
def infer_depth(
    midas,
    transform,
    x_rgb01: torch.Tensor,
) -> torch.Tensor:
    """
    x_rgb01 : (B,3,H,W) torch tensor in [0,1]
    returns  : depth01 (B,1,H,W) in [0,1]

    IMPORTANT :
    - DPT_Large exige une image numpy HxWx3
    - PAS un tensor torch directement
    """
    device = x_rgb01.device
    B, _, H, W = x_rgb01.shape

    depths = []

    for b in range(B):
        # --- torch -> numpy HWC uint8 ---
        img = (
            x_rgb01[b]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        img = (img * 255.0).astype(np.uint8)

        # --- MiDaS transform ---
        sample = transform(img)

        if isinstance(sample, dict) and "image" in sample:
            inp = sample["image"]
        else:
            inp = sample

        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp)

        if inp.ndim == 3:
            inp = inp.unsqueeze(0)

        inp = inp.to(device)

        # --- forward ---
        depth = midas(inp)              # (1,h,w)
        depth = depth.unsqueeze(1)      # (1,1,h,w)

        depth = F.interpolate(
            depth,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # --- normalize [0,1] ---
        depth = depth - depth.amin(dim=(2, 3), keepdim=True)
        depth = depth / (depth.amax(dim=(2, 3), keepdim=True) + 1e-6)

        depths.append(depth)

    return torch.cat(depths, dim=0)


def save_image(path: Path, x: np.ndarray, cmap: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    if cmap is None:
        plt.imshow(x)
    else:
        plt.imshow(x, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img_path = os.environ.get("M3_IMG", None)
    if img_path is None:
        raise RuntimeError("Please set M3_IMG=/path/to/image")

    print("Using image:", img_path)

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------
    x = load_rgb(img_path, size=256).to(device)

    # ------------------------------------------------------------
    # MiDaS depth (ALWAYS ON)
    # ------------------------------------------------------------
    midas, midas_transform = load_midas(device)
    depth01 = infer_depth(midas, midas_transform, x)

    # ------------------------------------------------------------
    # Run M3 (+ M4)
    # ------------------------------------------------------------
    out = run_m3(
        x_rgb01=x,
        depth01=depth01,
        num_slots=6,
        use_dino=True,
        dino_channels=24,
        geo_em_iters=6,
        geo_tau=0.20,
        w_barrier=6.0,
        w_feat=1.0,
        seed_mode="perceptual",
        inhibit=True,
        seed=0,
        run_m4=True,          # <<< M4 ACTIVÉ
        m4_tau=0.45,
    )

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------
    out_dir = Path("runs/m3_slots")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(img_path).stem

    save_image(
        out_dir / f"{base}_rgb.png",
        x[0].permute(1, 2, 0).cpu().numpy(),
    )

    save_image(
        out_dir / f"{base}_depth.png",
        depth01[0, 0].cpu().numpy(),
        cmap="gray",
    )

    save_image(
        out_dir / f"{base}_barrier.png",
        out.barrier[0, 0].cpu().numpy(),
        cmap="gray",
    )

    save_image(
        out_dir / f"{base}_s2_mean.png",
        out.s2[0].mean(dim=0).cpu().numpy(),
        cmap="gray",
    )

    save_image(
        out_dir / f"{base}_m32_kmeans.png",
        out.km_masks[0, 0].cpu().numpy(),
        cmap="gray",
    )

    save_image(
        out_dir / f"{base}_m34_labels.png",
        out.geo_labels[0].cpu().numpy(),
        cmap="tab20",
    )

    save_image(
        out_dir / f"{base}_m35_labels.png",
        out.m35_labels[0].cpu().numpy(),
        cmap="tab20",
    )

    # --- M4 (nouveau) ---
    if out.m4_labels is not None:
        save_image(
            out_dir / f"{base}_m4_labels.png",
            out.m4_labels[0].cpu().numpy(),
            cmap="tab20",
        )
        print("M4 regions:", out.m4_num_regions)

    print("Saved results to:", out_dir)


if __name__ == "__main__":
    main()

