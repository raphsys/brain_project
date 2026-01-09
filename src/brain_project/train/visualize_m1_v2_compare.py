from __future__ import annotations
import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception


def main():
    os.makedirs("./runs/visualize_v2_compare", exist_ok=True)
    device = torch.device("cpu")

    tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])

    ds = datasets.STL10("./data", split="train", download=True, transform=tf)
    x, _ = ds[7]
    x = x.unsqueeze(0).to(device)

    m1 = M1V1V2Perception(
        k_regions=8,
        use_depth=True,
        midas_work_res=256,
    ).to(device).eval()

    with torch.no_grad():
        out = m1(x)

    img = x[0].permute(1, 2, 0).cpu()
    boundary = out.boundary[0, 0].cpu()
    s1 = out.s1[0].cpu()
    arg = torch.argmax(s1, dim=0)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(out.depth[0, 0].cpu(), cmap="gray")
    plt.title("MiDaS depth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(boundary, cmap="gray")
    plt.title("V2 boundary (diffused)")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(arg, cmap="tab10")
    plt.title("S1 argmax")
    plt.axis("off")

    path = "./runs/visualize_v2_compare/v2_diffusion.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print("Saved:", path)


if __name__ == "__main__":
    main()

