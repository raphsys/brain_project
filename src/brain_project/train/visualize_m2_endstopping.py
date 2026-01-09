from __future__ import annotations

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception
from brain_project.modules.m2_grouping import EndStopping


def load_image(path: str, img_size: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    im = im.resize((img_size, img_size), resample=Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).float() / 255.0
    return x.permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def main():
    in_dir = os.environ.get("M2_IN_DIR", "./data/real_images")
    out_dir = os.environ.get("M2_OUT_DIR", "./runs/m2_endstopping")
    img_size = int(os.environ.get("M2_IMG_SIZE", "256"))
    midas_res = int(os.environ.get("M2_MIDAS_RES", "256"))

    os.makedirs(out_dir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(in_dir, e))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"No images found in {in_dir}")

    # pick one
    path = paths[0]
    x = load_image(path, img_size=img_size)

    # M1
    m1 = M1V1V2Perception(k_regions=8, use_depth=True, midas_work_res=midas_res).eval()
    out1 = m1(x)

    # M2 end-stopping
    m2 = EndStopping(n_bins=8, radius=7, edge_smooth=0).eval()
    out2 = m2(out1.boundary)  # boundary is already diffused in your current M1 setup

    img = x[0].permute(1, 2, 0).numpy()
    depth = out1.depth[0, 0].numpy()
    boundary = out1.boundary[0, 0].numpy()
    end_map = out2.end_map[0, 0].numpy()

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("MiDaS depth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(boundary, cmap="gray")
    plt.title("M1 V2 boundary")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(end_map, cmap="gray")
    plt.title("M2 end-stopping (end_map)")
    plt.axis("off")

    out_path = os.path.join(out_dir, "m2_endstopping.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved:", out_path)
    print("Input image:", path)


if __name__ == "__main__":
    main()

