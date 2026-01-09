from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v1_gabor import v1_gabor_energy
from .v2_grouping import v2_local_affinity
from .depth_midas import MiDaSSmallDepth


@dataclass
class M1V1V2Out:
    s1: torch.Tensor          # (B,K,H,W)
    depth: torch.Tensor       # (B,1,H,W)
    gabor_energy: torch.Tensor  # (B,1,H,W) energy summary
    boundary: torch.Tensor    # (B,1,H,W)
    feat_vec: torch.Tensor    # (B,F,H,W) features used for clustering


def _soft_kmeans(
    feat: torch.Tensor,
    k: int,
    iters: int = 10,
    temp: float = 0.15,
    seed: int = 123,
) -> torch.Tensor:
    """
    feat: (B,F,H,W) -> returns soft assignments S: (B,K,H,W)
    CPU-friendly, uses torch ops only.
    """
    torch.manual_seed(seed)
    B, Fch, H, W = feat.shape
    N = H * W

    # flatten pixels
    X = feat.view(B, Fch, N).transpose(1, 2).contiguous()  # (B,N,F)

    # normalize features
    X = X - X.mean(dim=1, keepdim=True)
    X = X / (X.std(dim=1, keepdim=True).clamp_min(1e-6))

    # init centroids by sampling pixels
    idx = torch.randint(low=0, high=N, size=(B, k), device=X.device)
    C = torch.gather(X, dim=1, index=idx.unsqueeze(-1).expand(B, k, Fch)).contiguous()  # (B,K,F)

    for _ in range(iters):
        # squared distances (B,N,K)
        # d(x,c) = ||x||^2 + ||c||^2 - 2 x·c
        x2 = (X * X).sum(dim=2, keepdim=True)          # (B,N,1)
        c2 = (C * C).sum(dim=2).unsqueeze(1)           # (B,1,K)
        xc = torch.bmm(X, C.transpose(1, 2))           # (B,N,K)
        d2 = (x2 + c2 - 2.0 * xc).clamp_min(0.0)

        # soft assignments
        S = torch.softmax(-d2 / temp, dim=2)           # (B,N,K)

        # update centroids
        denom = S.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-6)  # (B,K,1)
        C = torch.bmm(S.transpose(1, 2), X) / denom     # (B,K,F)

    S_img = S.transpose(1, 2).contiguous().view(B, k, H, W)  # (B,K,H,W)
    # ensure simplex
    S_img = S_img / (S_img.sum(dim=1, keepdim=True).clamp_min(1e-6))
    return S_img


class M1V1V2Perception(nn.Module):
    """
    M1 = V1 (Gabor energy) + V2 (local affinities/boundaries) + Depth (MiDaS-small)
         then soft-kmeans on a perceptual feature vector to produce soft regions S1.
    """
    def __init__(
        self,
        k_regions: int = 8,
        use_depth: bool = True,
        midas_work_res: int = 256,
        kmeans_iters: int = 12,
        kmeans_temp: float = 0.15,
    ):
        super().__init__()
        self.k_regions = k_regions
        self.use_depth = use_depth
        self.midas_work_res = midas_work_res
        self.kmeans_iters = kmeans_iters
        self.kmeans_temp = kmeans_temp

        self.depth_model = MiDaSSmallDepth(freeze=True) if use_depth else None

    def forward(self, x: torch.Tensor) -> M1V1V2Out:
        """
        x: (B,3,H,W) in [0,1]
        """
        B, C, H, W = x.shape

        # ---- Depth prior (MiDaS-small) ----
        if self.use_depth:
            depth = self.depth_model(x, work_res=self.midas_work_res).depth  # (B,1,H,W)
        else:
            depth = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)

        # ---- V1: Gabor energy ----
        v1 = v1_gabor_energy(x, orientations=8)
        gabor_e = v1.energy_sum  # (B,1,H,W)

        # ---- Feature vector for V2 affinity + clustering ----
        # Use: RGB (3) + depth (1) + gabor energy (1)
        feat_vec = torch.cat([x, depth, gabor_e], dim=1)  # (B,5,H,W)

        # ---- V2: boundaries from affinities ----
        v2 = v2_local_affinity(feat_vec, sigma=0.35)
        boundary = v2.boundary  # (B,1,H,W)
        boundary_diff = v2.boundary_diffused

        # ---- Clustering features (add boundary as extra signal) ----
        # boundary helps separate regions near edges
        cluster_feat = torch.cat([feat_vec, boundary], dim=1)  # (B,6,H,W)

        # ---- Soft k-means -> S1 ----
        s1 = _soft_kmeans(
            cluster_feat,
            k=self.k_regions,
            iters=self.kmeans_iters,
            temp=self.kmeans_temp,
        )

        return M1V1V2Out(
            s1=s1,
            depth=depth,
            gabor_energy=gabor_e,
            boundary=boundary,
            feat_vec=cluster_feat,
        )

