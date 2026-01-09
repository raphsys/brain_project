from __future__ import annotations

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception
from brain_project.modules.m2_grouping import CoCircularity, stabilize_regions


def load_image(path: str, img_size: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    im = im.resize((img_size, img_size), resample=Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).float() / 255.0
    return x.permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def main():
    in_dir = os.environ.get("M2_IN_DIR", "./data/real_images")
    out_dir = os.environ.get("M2_OUT_DIR", "./runs/m2_s2")
    img_size = int(os.environ.get("M2_IMG_SIZE", "256"))
    midas_res = int(os.environ.get("M2_MIDAS_RES", "256"))
    k_regions = int(os.environ.get("M2_K", "8"))

    os.makedirs(out_dir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(in_dir, e))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"No images found in {in_dir}")

    path = paths[0]
    x = load_image(path, img_size=img_size)

    # M1
    m1 = M1V1V2Perception(k_regions=k_regions, use_depth=True, midas_work_res=midas_res).eval()
    out1 = m1(x)

    # M2.2 co-circularity completion
    m2_c = CoCircularity(sigma_theta=0.35, sigma_dist=1.25, iters=10, alpha=0.20)
    out2 = m2_c(out1.boundary)

    # M2.3 region stabilization (S2)
    out3 = stabilize_regions(
        s1=out1.s1,
        barrier_m1=out1.boundary,
        completed_m2=out2.completed,
        depth=out1.depth,
        iters=10,
        alpha=0.35,
        beta_barrier=6.0,
        beta_depth=2.0,
    )

    img = x[0].permute(1, 2, 0).numpy()
    depth = out1.depth[0, 0].numpy()
    boundary = out1.boundary[0, 0].numpy()
    completed = out2.completed[0, 0].numpy()
    barrier = out3.barrier[0, 0].numpy()

    s1_arg = torch.argmax(out1.s1[0], dim=0).numpy()
    s2_arg = torch.argmax(out3.s2[0], dim=0).numpy()

    plt.figure(figsize=(20, 8))

    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("MiDaS depth")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(boundary, cmap="gray")
    plt.title("M1 boundary (diffused)")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(completed, cmap="gray")
    plt.title("M2 completed contours")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(s1_arg, cmap="tab10")
    plt.title("S1 argmax")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(barrier, cmap="gray")
    plt.title("Barrier used (max)")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(s2_arg, cmap="tab10")
    plt.title("S2 argmax (stabilized)")
    plt.axis("off")

    # show one region prob before/after (region 0)
    plt.subplot(2, 4, 8)
    plt.imshow(out3.s2[0, 0].numpy(), cmap="gray")
    plt.title("S2 region0 prob")
    plt.axis("off")

    out_path = os.path.join(out_dir, "m2_s2.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved:", out_path)
    print("Input image:", path)


if __name__ == "__main__":
    main()

