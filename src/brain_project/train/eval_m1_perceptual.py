from __future__ import annotations

import os
import json
import math
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from brain_project.data.loaders import LoaderSpec, build_dataset
from brain_project.modules.m1_perception import M1V1V2Perception
from brain_project.utils.metrics_m1 import compute_m1_metrics, M1Metrics


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _augment_light(x: torch.Tensor, seed: int = 123) -> torch.Tensor:
    """
    Light augmentation for stability test.
    x: (B,3,H,W) in [0,1]
    """
    set_seed(seed)
    B, C, H, W = x.shape

    # (1) slight resize jitter then back
    scale = random.uniform(0.90, 1.10)
    h2 = max(16, int(round(H * scale)))
    w2 = max(16, int(round(W * scale)))
    x2 = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    x2 = F.interpolate(x2, size=(H, W), mode="bilinear", align_corners=False)

    # (2) mild brightness/contrast jitter
    brightness = random.uniform(0.95, 1.05)
    contrast = random.uniform(0.95, 1.05)
    mean = x2.mean(dim=(2, 3), keepdim=True)
    x3 = (x2 - mean) * contrast + mean
    x3 = (x3 * brightness).clamp(0.0, 1.0)

    # (3) occasional horizontal flip
    if random.random() < 0.5:
        x3 = torch.flip(x3, dims=[3])

    return x3


def _save_panel(
    out_path: str,
    img: np.ndarray,
    depth: np.ndarray,
    gabor: np.ndarray,
    boundary: np.ndarray,
    s1: torch.Tensor,
):
    """
    Save a consistent panel for qualitative inspection.
    s1: (K,H,W)
    """
    K, H, W = s1.shape
    arg = torch.argmax(s1, dim=0).cpu().numpy()

    cols = 4
    rows = int(math.ceil((K + 5) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))

    # Original
    plt.subplot(rows, cols, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    # Depth
    plt.subplot(rows, cols, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("MiDaS depth (norm)")
    plt.axis("off")

    # Gabor
    plt.subplot(rows, cols, 3)
    plt.imshow(gabor, cmap="gray")
    plt.title("V1 Gabor energy")
    plt.axis("off")

    # Boundary
    plt.subplot(rows, cols, 4)
    plt.imshow(boundary, cmap="gray")
    plt.title("V2 boundary strength")
    plt.axis("off")

    # Argmax
    plt.subplot(rows, cols, 5)
    plt.imshow(arg, cmap="tab10")
    plt.title("S1 argmax (regions)")
    plt.axis("off")

    # Regions
    for k in range(K):
        plt.subplot(rows, cols, 6 + k)
        plt.imshow(s1[k].cpu().numpy(), cmap="gray")
        plt.title(f"Region {k}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def eval_dataset(
    name: str,
    split: str,
    n_images: int,
    img_size: int,
    k_regions: int,
    out_root: str,
    midas_work_res: int,
):
    device = torch.device("cpu")

    ds = build_dataset(LoaderSpec(name=name, split=split, img_size=img_size))
    n_images = min(n_images, len(ds))

    out_dir = os.path.join(out_root, f"{name}_{split}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)

    m1 = M1V1V2Perception(
        k_regions=k_regions,
        use_depth=True,
        midas_work_res=midas_work_res,
        kmeans_iters=12,
        kmeans_temp=0.15,
    ).to(device).eval()

    metrics: List[M1Metrics] = []

    for i in tqdm(range(n_images), desc=f"[M1 eval] {name}/{split}"):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(device)  # (1,3,H,W)

        out = m1(x)
        s1 = out.s1
        depth = out.depth
        gabor = out.gabor_energy
        boundary = out.boundary

        # Aug stability
        x_aug = _augment_light(x, seed=123 + i)
        out_aug = m1(x_aug)
        s1_aug = out_aug.s1

        m = compute_m1_metrics(x=x, s1=s1, boundary=boundary, s1_aug=s1_aug)
        metrics.append(m)

        # Save a few qualitative panels
        if i < 24:
            img = x[0].permute(1, 2, 0).cpu().numpy()
            panel_path = os.path.join(out_dir, "panels", f"panel_{i:04d}.png")
            _save_panel(
                panel_path,
                img=img,
                depth=depth[0, 0].cpu().numpy(),
                gabor=gabor[0, 0].cpu().numpy(),
                boundary=boundary[0, 0].cpu().numpy(),
                s1=s1[0],
            )

    # Aggregate metrics
    def mean_std(vals: List[float]) -> Tuple[float, float]:
        a = np.array(vals, dtype=np.float64)
        return float(a.mean()), float(a.std())

    ent_m, ent_s = mean_std([m.entropy_mean for m in metrics])
    entstd_m, entstd_s = mean_std([m.entropy_std for m in metrics])
    gini_m, gini_s = mean_std([m.region_area_gini for m in metrics])
    corr_m, corr_s = mean_std([m.boundary_grad_corr for m in metrics])
    kl_m, kl_s = mean_std([m.stability_kl for m in metrics])

    summary = {
        "dataset": name,
        "split": split,
        "n_images": n_images,
        "img_size": img_size,
        "k_regions": k_regions,
        "midas_work_res": midas_work_res,
        "metrics_mean_std": {
            "entropy_mean": [ent_m, ent_s],
            "entropy_std": [entstd_m, entstd_s],
            "region_area_gini": [gini_m, gini_s],
            "boundary_grad_corr": [corr_m, corr_s],
            "stability_kl": [kl_m, kl_s],
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also save per-image metrics
    with open(os.path.join(out_dir, "metrics_per_image.jsonl"), "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(asdict(m)) + "\n")

    print(f"\nSaved results to: {out_dir}")
    print(json.dumps(summary["metrics_mean_std"], indent=2))


def main():
    out_root = "./runs/m1_eval"
    os.makedirs(out_root, exist_ok=True)

    # Defaults: modest for CPU
    n_images = 100
    img_size = 160          # a bit higher than 128 for richer structure
    k_regions = 8
    midas_work_res = 256    # reduce to 192 if too slow

    # CIFAR-100
    eval_dataset(
        name="cifar100",
        split="train",
        n_images=n_images,
        img_size=img_size,
        k_regions=k_regions,
        out_root=out_root,
        midas_work_res=midas_work_res,
    )

    # STL-10
    eval_dataset(
        name="stl10",
        split="train",
        n_images=n_images,
        img_size=img_size,
        k_regions=k_regions,
        out_root=out_root,
        midas_work_res=midas_work_res,
    )


if __name__ == "__main__":
    main()

