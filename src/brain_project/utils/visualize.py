from __future__ import annotations

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from brain_project.modules.m1_perception import M1PerceptionInherited


def visualize_s1(
    model: M1PerceptionInherited,
    dataset: str = "cifar10",
    data_root: str = "./data",
    index: int = 0,
    img_size: int = 128,
    out_dir: str = "./runs/visualize_s1",
):
    """
    Visualize soft regions S1 for a single image.
    """

    device = torch.device("cpu")
    model = model.to(device).eval()

    # -------- dataset --------
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    if dataset == "cifar10":
        ds = datasets.CIFAR10(data_root, train=True, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    x, y = ds[index]
    x = x.unsqueeze(0).to(device)  # (1,3,H,W)

    # -------- forward --------
    with torch.no_grad():
        out = model(x)
        s1 = out.s1[0]  # (K,H,W)

    K, H, W = s1.shape

    # -------- plots --------
    os.makedirs(out_dir, exist_ok=True)

    fig_cols = 4
    fig_rows = int(np.ceil((K + 2) / fig_cols))
    plt.figure(figsize=(4 * fig_cols, 4 * fig_rows))

    # Original image
    plt.subplot(fig_rows, fig_cols, 1)
    plt.imshow(x[0].permute(1, 2, 0).cpu())
    plt.title("Original image")
    plt.axis("off")

    # Argmax map
    plt.subplot(fig_rows, fig_cols, 2)
    arg = torch.argmax(s1, dim=0).cpu().numpy()
    plt.imshow(arg, cmap="tab10")
    plt.title("S1 argmax (regions)")
    plt.axis("off")

    # Each region
    for k in range(K):
        plt.subplot(fig_rows, fig_cols, 3 + k)
        plt.imshow(s1[k].cpu(), cmap="gray")
        plt.title(f"Region {k}")
        plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"s1_regions_idx_{index}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved visualization to: {out_path}")

