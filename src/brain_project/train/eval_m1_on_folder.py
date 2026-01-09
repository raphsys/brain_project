from __future__ import annotations

import os
import glob
import json
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception
from brain_project.utils.metrics_m1 import compute_m1_metrics


def load_image(path: str, img_size: int) -> torch.Tensor:
    """
    Returns x: (1,3,H,W) float in [0,1]
    """
    im = Image.open(path).convert("RGB")
    im = im.resize((img_size, img_size), resample=Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).float() / 255.0  # (H,W,3)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()    # (1,3,H,W)
    return x


def augment_light(x: torch.Tensor, seed: int = 123) -> torch.Tensor:
    torch.manual_seed(seed)
    B, C, H, W = x.shape

    scale = float(torch.empty(1).uniform_(0.90, 1.10).item())
    h2 = max(16, int(round(H * scale)))
    w2 = max(16, int(round(W * scale)))
    x2 = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    x2 = F.interpolate(x2, size=(H, W), mode="bilinear", align_corners=False)

    brightness = float(torch.empty(1).uniform_(0.95, 1.05).item())
    contrast = float(torch.empty(1).uniform_(0.95, 1.05).item())
    mean = x2.mean(dim=(2, 3), keepdim=True)
    x3 = (x2 - mean) * contrast + mean
    x3 = (x3 * brightness).clamp(0.0, 1.0)

    if float(torch.rand(1).item()) < 0.5:
        x3 = torch.flip(x3, dims=[3])

    return x3


def save_panel(out_path: str, x: torch.Tensor, out) -> None:
    """
    x: (1,3,H,W), out: M1V1V2Out
    """
    img = x[0].permute(1, 2, 0).cpu().numpy()
    depth = out.depth[0, 0].cpu().numpy()
    boundary = out.boundary[0, 0].cpu().numpy()
    s1 = out.s1[0].cpu()  # (K,H,W)
    arg = torch.argmax(s1, dim=0).numpy()

    K = s1.shape[0]
    cols = 4
    rows = int(np.ceil((K + 4) / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))

    plt.subplot(rows, cols, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(rows, cols, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("MiDaS depth")
    plt.axis("off")

    plt.subplot(rows, cols, 3)
    plt.imshow(boundary, cmap="gray")
    plt.title("V2 boundary (diffused)")
    plt.axis("off")

    plt.subplot(rows, cols, 4)
    plt.imshow(arg, cmap="tab10")
    plt.title("S1 argmax")
    plt.axis("off")

    for k in range(K):
        plt.subplot(rows, cols, 5 + k)
        plt.imshow(s1[k].numpy(), cmap="gray")
        plt.title(f"Region {k}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def main():
    in_dir = os.environ.get("M1_IN_DIR", "./data/real_images")
    out_dir = os.environ.get("M1_OUT_DIR", "./runs/m1_real")
    img_size = int(os.environ.get("M1_IMG_SIZE", "256"))
    n_images = int(os.environ.get("M1_N", "50"))

    k_regions = int(os.environ.get("M1_K", "8"))
    midas_work_res = int(os.environ.get("M1_MIDAS_RES", "256"))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(in_dir, e)))
    paths = sorted(paths)[:n_images]

    if not paths:
        raise SystemExit(f"No images found in {in_dir} (jpg/jpeg/png/webp).")

    device = torch.device("cpu")

    m1 = M1V1V2Perception(
        k_regions=k_regions,
        use_depth=True,
        midas_work_res=midas_work_res,
        kmeans_iters=12,
        kmeans_temp=0.15,
    ).to(device).eval()

    per_image = []
    for i, p in enumerate(tqdm(paths, desc="[M1 real]")):
        x = load_image(p, img_size=img_size).to(device)

        out = m1(x)
        x_aug = augment_light(x, seed=1000 + i)
        out_aug = m1(x_aug)

        m = compute_m1_metrics(x=x, s1=out.s1, boundary=out.boundary, s1_aug=out_aug.s1)
        rec = {
            "path": p,
            "metrics": asdict(m),
        }
        per_image.append(rec)

        panel_path = os.path.join(out_dir, "panels", f"panel_{i:04d}.png")
        save_panel(panel_path, x, out)

    # summarize
    def mean_std(key: str) -> Tuple[float, float]:
        vals = [r["metrics"][key] for r in per_image]
        a = np.array(vals, dtype=np.float64)
        return float(a.mean()), float(a.std())

    summary = {
        "in_dir": in_dir,
        "n_images": len(per_image),
        "img_size": img_size,
        "k_regions": k_regions,
        "midas_work_res": midas_work_res,
        "metrics_mean_std": {
            "entropy_mean": list(mean_std("entropy_mean")),
            "entropy_std": list(mean_std("entropy_std")),
            "region_area_gini": list(mean_std("region_area_gini")),
            "boundary_grad_corr": list(mean_std("boundary_grad_corr")),
            "stability_kl": list(mean_std("stability_kl")),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "per_image.jsonl"), "w", encoding="utf-8") as f:
        for r in per_image:
            f.write(json.dumps(r) + "\n")

    print("\nSaved panels to:", os.path.join(out_dir, "panels"))
    print("Saved summary to:", os.path.join(out_dir, "summary.json"))
    print(json.dumps(summary["metrics_mean_std"], indent=2))


if __name__ == "__main__":
    main()

