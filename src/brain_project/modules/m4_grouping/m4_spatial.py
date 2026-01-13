# FILE: src/brain_project/modules/m4_grouping/m4_spatial.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class M4SpatialOut:
    labels: torch.Tensor          # (B,H,W) labels after grouping
    num_regions: int
    mapping: Dict[int, int]       # old_root -> new_id


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _region_means(
    labels: torch.Tensor,          # (H,W) ints [0..R-1]
    feat: torch.Tensor,            # (C,H,W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return region mean features: (R,C)
    """
    H, W = labels.shape
    device = labels.device
    R = int(labels.max().item()) + 1

    means = torch.zeros((R, feat.shape[0]), device=device, dtype=feat.dtype)
    counts = torch.zeros((R,), device=device, dtype=feat.dtype)

    flat_lab = labels.view(-1)                       # (HW,)
    flat_feat = feat.view(feat.shape[0], -1).t()     # (HW,C)

    for r in range(R):
        mask = flat_lab == r
        if mask.any():
            fr = flat_feat[mask]                     # (n,C)
            means[r] = fr.mean(dim=0)
            counts[r] = float(mask.sum().item())
        else:
            means[r] = 0.0
            counts[r] = 0.0

    return means


def _region_centroids(
    labels: torch.Tensor,          # (H,W)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return normalized centroids in [-1,1]: (R,2) as (cx,cy)
    """
    H, W = labels.shape
    device = labels.device
    R = int(labels.max().item()) + 1

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij",
    )
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    flat_lab = labels.view(-1)

    cent = torch.zeros((R, 2), device=device)
    for r in range(R):
        mask = flat_lab == r
        if mask.any():
            cent[r, 0] = xx[mask].mean()
            cent[r, 1] = yy[mask].mean()
        else:
            cent[r] = 0.0

    return cent


def _adjacent_pairs_4n(labels: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Region adjacency from 4-neighborhood, returns unique pairs (i,j) with i<j.
    """
    H, W = labels.shape
    pairs = set()

    # vertical edges
    a = labels[:-1, :]
    b = labels[1:, :]
    diff = a != b
    if diff.any():
        pa = a[diff].flatten().tolist()
        pb = b[diff].flatten().tolist()
        for i, j in zip(pa, pb):
            if i != j:
                pairs.add((min(i, j), max(i, j)))

    # horizontal edges
    a = labels[:, :-1]
    b = labels[:, 1:]
    diff = a != b
    if diff.any():
        pa = a[diff].flatten().tolist()
        pb = b[diff].flatten().tolist()
        for i, j in zip(pa, pb):
            if i != j:
                pairs.add((min(i, j), max(i, j)))

    return list(pairs)


@torch.no_grad()
def m4_spatial_grouping(
    labels_m3: torch.Tensor,       # (B,H,W) int labels (proto-objects from M3)
    sem: torch.Tensor,             # (B,Dsem,H,W) semantic features (e.g. DINO proj) OR zeros
    per: torch.Tensor,             # (B,Dper,H,W) perceptual features (e.g. s2_low) OR zeros
    lam_sem: float = 0.45,
    lam_per: float = 0.45,
    lam_cent: float = 0.10,
    tau: float = 0.45,
    eps: float = 1e-6,
) -> M4SpatialOut:
    """
    M4 (spatial): merge ONLY adjacent regions if combined distance < tau.
    Distance = lam_sem*(1-cos) + lam_per*(1-cos) + lam_cent*L2(centroids)
    """

    assert labels_m3.ndim == 3, "labels_m3 must be (B,H,W)"
    B, H, W = labels_m3.shape
    device = labels_m3.device

    assert sem.ndim == 4 and per.ndim == 4, "sem/per must be (B,D,H,W)"
    assert sem.shape[0] == B and sem.shape[-2:] == (H, W)
    assert per.shape[0] == B and per.shape[-2:] == (H, W)

    # (for now) do per-image union-find (B usually 1 here)
    out_labels = torch.empty_like(labels_m3)

    mapping_all: Dict[int, int] = {}
    num_total = 0

    for bb in range(B):
        labels = labels_m3[bb]
        R = int(labels.max().item()) + 1
        uf = _UnionFind(R)

        sem_means = _region_means(labels, sem[bb], eps=eps)   # (R,Dsem)
        per_means = _region_means(labels, per[bb], eps=eps)   # (R,Dper)
        cents = _region_centroids(labels, eps=eps)            # (R,2)

        # normalize for cosine
        sem_means = sem_means / (sem_means.norm(dim=1, keepdim=True) + eps)
        per_means = per_means / (per_means.norm(dim=1, keepdim=True) + eps)

        pairs = _adjacent_pairs_4n(labels)

        for i, j in pairs:
            # cosine distances (1 - cos)
            d_sem = 1.0 - (sem_means[i] * sem_means[j]).sum()
            d_per = 1.0 - (per_means[i] * per_means[j]).sum()
            d_cent = (cents[i] - cents[j]).norm()

            d = lam_sem * d_sem + lam_per * d_per + lam_cent * d_cent
            if float(d.item()) < tau:
                uf.union(i, j)

        # compress + relabel contiguous
        roots = [uf.find(i) for i in range(R)]
        root_to_new: Dict[int, int] = {}
        next_id = 0
        for r in roots:
            if r not in root_to_new:
                root_to_new[r] = next_id
                next_id += 1

        # rewrite pixel labels
        new_lab = torch.empty_like(labels)
        for old in range(R):
            root = uf.find(old)
            new_id = root_to_new[root]
            new_lab[labels == old] = new_id

        out_labels[bb] = new_lab

        # store mapping (debug)
        for root, new_id in root_to_new.items():
            mapping_all[int(root) + num_total] = int(new_id) + num_total

        num_total += next_id

    return M4SpatialOut(
        labels=out_labels,
        num_regions=int(out_labels.max().item()) + 1,
        mapping=mapping_all,
    )

