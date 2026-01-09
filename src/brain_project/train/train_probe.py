from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from brain_project.modules.m1_perception import M1PerceptionInherited


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def few_shot_subset(dataset, n_per_class: int, num_classes: int = 10, seed: int = 123):
    set_seed(seed)
    indices_by_class = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        _, y = dataset[i]
        indices_by_class[y].append(i)

    chosen = []
    for c in range(num_classes):
        idxs = indices_by_class[c]
        random.shuffle(idxs)
        chosen.extend(idxs[:n_per_class])

    return Subset(dataset, chosen)


# -----------------------
# Embedding extractors
# -----------------------
@torch.no_grad()
def extract_embeddings_baseline(m1, loader, device):
    """
    Baseline: global average pooling of backbone features.
    """
    X, Y = [], []
    m1.eval()
    for x, y in loader:
        x = x.to(device)
        out = m1(x)
        feat = out.feat                      # (B, C, Hf, Wf)
        emb = feat.mean(dim=(2, 3))          # GAP -> (B, C)
        X.append(emb.cpu())
        Y.append(y)
    return torch.cat(X), torch.cat(Y)


@torch.no_grad()
def extract_embeddings_s1(m1, loader, device):
    """
    Ours: S1-guided regional pooling.
    For each region k:
      e_k = sum_{p}( S1_k(p) * feat(p) ) / sum_{p} S1_k(p)
    Then concatenate all regions: (B, K*C)
    """
    X, Y = [], []
    m1.eval()
    for x, y in loader:
        x = x.to(device)
        out = m1(x)
        feat = out.feat          # (B, C, Hf, Wf)
        s1   = out.s1            # (B, K, H, W)

        # downsample S1 to feature resolution
        s1f = torch.nn.functional.interpolate(
            s1, size=feat.shape[-2:], mode="bilinear", align_corners=False
        )                          # (B, K, Hf, Wf)

        B, K, Hf, Wf = s1f.shape
        C = feat.shape[1]

        # compute regional embeddings
        regs = []
        for k in range(K):
            w = s1f[:, k:k+1]                          # (B,1,Hf,Wf)
            num = (feat * w).sum(dim=(2, 3))           # (B,C)
            den = w.sum(dim=(2, 3)).clamp_min(1e-6)    # (B,1)
            ek = num / den                             # (B,C)
            regs.append(ek)

        emb = torch.cat(regs, dim=1)                    # (B, K*C)
        X.append(emb.cpu())
        Y.append(y)
    return torch.cat(X), torch.cat(Y)


# -----------------------
# Linear probe
# -----------------------
def train_linear_probe(Xtr, Ytr, Xte, Yte, num_classes=10, epochs=200, lr=1e-2):
    device = torch.device("cpu")
    model = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xtr.to(device))
        loss = loss_fn(logits, Ytr.to(device))
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(Xte.to(device))
        acc = (logits.argmax(dim=1).cpu() == Yte).float().mean().item()
    return acc


# -----------------------
# Main experiment
# -----------------------
def main():
    device = torch.device("cpu")
    set_seed(123)

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=tf)

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # M1 perception inherited (frozen)
    m1 = M1PerceptionInherited(
        k_regions=8,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    shots = [1, 5, 10, 50]
    print("\n=== Linear Probe Few-Shot Results ===")

    for n in shots:
        fs_ds = few_shot_subset(train_ds, n_per_class=n, seed=123)
        fs_loader = DataLoader(fs_ds, batch_size=32, shuffle=True, num_workers=2)

        # Baseline
        Xtr_b, Ytr = extract_embeddings_baseline(m1, fs_loader, device)
        Xte_b, Yte = extract_embeddings_baseline(m1, test_loader, device)
        acc_b = train_linear_probe(Xtr_b, Ytr, Xte_b, Yte)

        # Ours (S1-guided)
        Xtr_s, _ = extract_embeddings_s1(m1, fs_loader, device)
        Xte_s, _ = extract_embeddings_s1(m1, test_loader, device)
        acc_s = train_linear_probe(Xtr_s, Ytr, Xte_s, Yte)

        print(f"{n:>3} shot(s) | Baseline: {acc_b*100:5.1f}% | S1-guided: {acc_s*100:5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()

