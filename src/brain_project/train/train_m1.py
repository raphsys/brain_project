from __future__ import annotations

import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from brain_project.modules.m1_perception import M1PerceptionInherited


def main():
    device = torch.device("cpu")

    # Small input for CPU speed
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

    m1 = M1PerceptionInherited(k_regions=8, pretrained=True, freeze_backbone=True).to(device)
    m1.eval()

    x, _ = next(iter(dl))
    x = x.to(device)

    t0 = time.time()
    with torch.no_grad():
        out = m1(x)
    dt = time.time() - t0

    print("M1 output:")
    print("  s1:", tuple(out.s1.shape), "sum_k:", out.s1.sum(dim=1).mean().item())
    print("  feat:", tuple(out.feat.shape))
    print("  logits:", tuple(out.logits.shape))
    print(f"  forward time (batch=16): {dt:.3f}s")


if __name__ == "__main__":
    main()

