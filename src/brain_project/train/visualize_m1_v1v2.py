from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception


def main():
    os.makedirs("./runs/visualize_m1_v1v2", exist_ok=True)
    device = torch.device("cpu")

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    ds = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    x, y = ds[3]
    x = x.unsqueeze(0).to(device)

    m1 = M1V1V2Perception(
        k_regions=8,
        use_depth=True,
        midas_work_res=256,
        kmeans_iters=12,
        kmeans_temp=0.15,
    ).to(device).eval()

    with torch.no_grad():
        out = m1(x)

    img = x[0].permute(1, 2, 0).cpu().numpy()
    depth = out.depth[0, 0].cpu().numpy()
    gabor = out.gabor_energy[0, 0].cpu().numpy()
    boundary = out.boundary[0, 0].cpu().numpy()
    s1 = out.s1[0].cpu()
    arg = torch.argmax(s1, dim=0).numpy()

    K = s1.shape[0]
    cols = 4
    rows = int(np.ceil((K + 5) / cols))

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

    # Gabor energy
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
        plt.imshow(s1[k].numpy(), cmap="gray")
        plt.title(f"Region {k}")
        plt.axis("off")

    plt.tight_layout()
    out_path = "./runs/visualize_m1_v1v2/m1_v1v2_idx_3.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

