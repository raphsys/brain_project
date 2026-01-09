import os
import numpy as np
import torch

from brain_project.modules.m3_invariance.slots_kmeans import soft_kmeans_slots


def slot_entropy(masks, eps=1e-6):
    # masks: (B,S,H,W) in [0,1], sum over slots not necessarily 1
    m = masks.clamp(eps, 1.0)
    p = m / (m.sum(dim=1, keepdim=True) + eps)
    ent = -(p * p.log()).sum(dim=1).mean().item()
    return ent


def mean_pairwise_iou(masks, thr=0.5, eps=1e-6):
    # masks: (B,S,H,W)
    B, S, H, W = masks.shape
    mb = (masks > thr).float()
    ious = []
    for b in range(B):
        for i in range(S):
            for j in range(i + 1, S):
                inter = (mb[b, i] * mb[b, j]).sum()
                union = ((mb[b, i] + mb[b, j]) > 0).float().sum()
                iou = (inter / (union + eps)).item()
                ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def main():
    cache = "./runs/m3_cache/test.pt"
    if not os.path.exists(cache):
        raise RuntimeError("Missing cache. Run train_m3 to build caches first.")

    S2 = torch.load(cache)  # (N,K,H,W)
    # take a small subset for speed
    S2 = S2[:32]

    slots = soft_kmeans_slots(S2, num_slots=6, iters=12, tau=0.25, add_coords=True)

    ent = slot_entropy(slots.masks)
    iou = mean_pairwise_iou(slots.masks, thr=0.5)

    print("=== M3.2 Slots metrics ===")
    print("entropy (higher=more spread):", ent)
    print("pairwise IoU (lower=better diversity):", iou)


if __name__ == "__main__":
    main()

