# FILE: src/brain_project/modules/m3_invariance/m4_grouping.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F


# ============================================================
# Union-Find (Disjoint Set)
# ============================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ============================================================
# Outputs
# ============================================================

@dataclass
class M4Out:
    labels: torch.Tensor                 # (B,H,W)
    num_regions_before: int
    num_regions_after: int
    edges_considered: int
    merges_done: int
    debug_region_feats: Optional[Dict[str, torch.Tensor]] = None


# ============================================================
# RAG + region stats
# ============================================================

def _compute_adjacency(labels: torch.Tensor) -> torch.Tensor:
    """
    labels: (H,W) int64
    returns edges: (E,2) unique undirected adjacency pairs
    """
    H, W = labels.shape
    lab = labels

    # horizontal neighbors
    a = lab[:, :-1]
    b = lab[:, 1:]
    mask = a != b
    e1 = torch.stack([a[mask], b[mask]], dim=1) if mask.any() else lab.new_zeros((0,2))

    # vertical neighbors
    a = lab[:-1, :]
    b = lab[1:, :]
    mask = a != b
    e2 = torch.stack([a[mask], b[mask]], dim=1) if mask.any() else lab.new_zeros((0,2))

    edges = torch.cat([e1, e2], dim=0)
    if edges.numel() == 0:
        return edges

    # undirected canonical ordering
    u = torch.minimum(edges[:, 0], edges[:, 1])
    v = torch.maximum(edges[:, 0], edges[:, 1])
    edges = torch.stack([u, v], dim=1)

    # unique
    edges = torch.unique(edges, dim=0)
    return edges


def _region_means(labels: torch.Tensor, x: torch.Tensor, num_regions: int, eps: float = 1e-6) -> torch.Tensor:
    """
    labels: (H,W) int64 in [0, R-1]
    x: (C,H,W) float
    returns: (R,C) mean per region
    """
    H, W = labels.shape
    C = x.shape[0]

    flat_lab = labels.view(-1)  # (HW,)
    flat_x = x.view(C, -1).t()  # (HW,C)

    sums = torch.zeros((num_regions, C), device=x.device, dtype=x.dtype)
    cnts = torch.zeros((num_regions, 1), device=x.device, dtype=x.dtype)

    sums.index_add_(0, flat_lab, flat_x)
    ones = torch.ones((flat_lab.shape[0], 1), device=x.device, dtype=x.dtype)
    cnts.index_add_(0, flat_lab, ones)

    return sums / (cnts + eps)


def _region_counts(labels: torch.Tensor, num_regions: int) -> torch.Tensor:
    flat = labels.view(-1)
    cnt = torch.bincount(flat, minlength=num_regions).to(labels.device)
    return cnt  # (R,)


# ============================================================
# Main M4: lightweight semantic merging
# ============================================================

@torch.no_grad()
def m4_group_regions(
    labels_m3: torch.Tensor,     # (B,H,W) long
    x_rgb01: torch.Tensor,       # (B,3,H,W) float
    depth01: torch.Tensor,       # (B,1,H,W) float
    s2: Optional[torch.Tensor] = None,  # (B,C,H,W) optional
    # weights
    w_rgb: float = 1.0,
    w_depth: float = 1.0,
    w_s2: float = 0.3,
    w_size: float = 0.05,
    # thresholds
    merge_thresh: float = 0.25,
    min_region_area: int = 80,
    max_passes: int = 6,
    eps: float = 1e-6,
    return_debug: bool = True,
) -> M4Out:
    """
    Graph-based region merging on adjacency graph (RAG).
    Very lightweight "semantic-ish" grouping.

    merge cost between adjacent regions i,j:
      cost = w_rgb*||rgb_i-rgb_j|| + w_depth*|d_i-d_j| + w_s2*||s2_i-s2_j|| + w_size*penalty_small
    merge if cost < merge_thresh.
    """
    assert labels_m3.ndim == 3
    B, H, W = labels_m3.shape
    device = labels_m3.device

    # We'll do per-image for clarity (B=1 in your use-case anyway)
    all_out = []
    total_merges = 0
    total_edges = 0
    before_total = 0
    after_total = 0

    for b in range(B):
        labels = labels_m3[b].clone()

        # reindex labels to [0..R-1] (compact)
        uniq = torch.unique(labels)
        remap = torch.empty((int(uniq.max().item()) + 1,), device=device, dtype=torch.long).fill_(-1)
        remap[uniq] = torch.arange(uniq.numel(), device=device)
        labels = remap[labels]
        R = int(uniq.numel())
        before_total += R

        x = x_rgb01[b]
        d = depth01[b]

        s2b = s2[b] if (s2 is not None) else None

        merges_done = 0

        for _pass in range(max_passes):
            # stats
            rgb_mu = _region_means(labels, x, R, eps=eps)            # (R,3)
            d_mu   = _region_means(labels, d, R, eps=eps).squeeze(1) # (R,)
            cnt    = _region_counts(labels, R).float()               # (R,)

            if s2b is not None:
                # reduce s2 dimension if huge (optional). Here: use mean over channels blocks.
                # Keep it simple: per-region mean over all channels
                s2_mu = _region_means(labels, s2b, R, eps=eps)       # (R,C)
            else:
                s2_mu = None

            # adjacency edges
            edges = _compute_adjacency(labels)
            E = int(edges.shape[0])
            total_edges += E
            if E == 0 or R <= 1:
                break

            # compute costs for all edges
            i = edges[:, 0]
            j = edges[:, 1]

            rgb_cost = torch.linalg.norm(rgb_mu[i] - rgb_mu[j], dim=1)  # (E,)
            d_cost = torch.abs(d_mu[i] - d_mu[j])                       # (E,)

            if s2_mu is not None:
                s2_cost = torch.linalg.norm(s2_mu[i] - s2_mu[j], dim=1)
            else:
                s2_cost = torch.zeros_like(rgb_cost)

            # penalty if one of regions is too small => encourage merge
            small_i = (cnt[i] < float(min_region_area)).float()
            small_j = (cnt[j] < float(min_region_area)).float()
            size_bonus = torch.maximum(small_i, small_j)  # 1 if either small

            cost = (
                w_rgb * rgb_cost
                + w_depth * d_cost
                + w_s2 * s2_cost
                - w_size * size_bonus
            )

            # pick candidate merges
            keep = cost < merge_thresh
            cand = edges[keep]
            if cand.numel() == 0:
                break

            # union-find merges
            uf = UnionFind(R)
            # sort candidates by increasing cost (greedy stable)
            order = torch.argsort(cost[keep])
            cand = cand[order]

            merged_this_pass = 0
            for a, bb2 in cand.tolist():
                if uf.union(a, bb2):
                    merged_this_pass += 1

            if merged_this_pass == 0:
                break

            merges_done += merged_this_pass

            # apply mapping to new compact labels
            roots = torch.tensor([uf.find(k) for k in range(R)], device=device, dtype=torch.long)
            # compress roots to 0..R'-1
            uniq_roots = torch.unique(roots)
            new_id = torch.empty((int(uniq_roots.max().item()) + 1,), device=device, dtype=torch.long).fill_(-1)
            new_id[uniq_roots] = torch.arange(uniq_roots.numel(), device=device)

            labels = new_id[roots[labels]]
            R = int(uniq_roots.numel())

        after_total += R
        total_merges += merges_done

        all_out.append(labels)

    labels_out = torch.stack(all_out, dim=0)

    debug = None
    if return_debug:
        debug = {
            "labels_m4": labels_out,
        }

    return M4Out(
        labels=labels_out,
        num_regions_before=before_total,
        num_regions_after=after_total,
        edges_considered=total_edges,
        merges_done=total_merges,
        debug_region_feats=debug,
    )

