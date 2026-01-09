from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class M1Metrics:
    entropy_mean: float
    entropy_std: float
    region_area_gini: float
    boundary_grad_corr: float
    stability_kl: float


def _entropy_per_pixel(s1: torch.Tensor) -> torch.Tensor:
    """
    s1: (B,K,H,W) on simplex
    returns: (B,H,W) entropy
    """
    eps = 1e-8
    p = s1.clamp_min(eps)
    h = -(p * p.log()).sum(dim=1)  # (B,H,W)
    return h


def entropy_stats(s1: torch.Tensor) -> Tuple[float, float]:
    h = _entropy_per_pixel(s1)             # (B,H,W)
    return float(h.mean().item()), float(h.std().item())


def region_area_gini(s1: torch.Tensor) -> float:
    """
    Compute Gini coefficient of hard region areas, averaged over batch.
    Lower gini => more balanced regions. Higher => one region dominates.
    """
    B, K, H, W = s1.shape
    hard = torch.argmax(s1, dim=1)  # (B,H,W)
    g_list = []
    for b in range(B):
        counts = torch.bincount(hard[b].view(-1), minlength=K).float()
        x = counts / counts.sum().clamp_min(1.0)  # proportions
        # Gini for proportions:
        # G = sum_i sum_j |xi - xj| / (2 K sum_i xi) ; but sum_i xi = 1
        diff = torch.abs(x.unsqueeze(0) - x.unsqueeze(1)).sum()
        g = diff / (2.0 * K)
        g_list.append(g)
    return float(torch.stack(g_list).mean().item())


def _image_luminance(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W) in [0,1]
    lum = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).clamp(0, 1)
    return lum


def boundary_gradient_correlation(x: torch.Tensor, boundary: torch.Tensor) -> float:
    """
    x: (B,3,H,W), boundary: (B,1,H,W)
    Compute Pearson correlation between boundary strength and image gradient magnitude.
    """
    lum = _image_luminance(x)  # (B,1,H,W)

    # Sobel filters
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=lum.dtype, device=lum.device).view(1,1,3,3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=lum.dtype, device=lum.device).view(1,1,3,3)

    gx = F.conv2d(lum, kx, padding=1)
    gy = F.conv2d(lum, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-8)  # (B,1,H,W)

    b = boundary
    g = grad

    # Flatten
    b = b.reshape(b.shape[0], -1)
    g = g.reshape(g.shape[0], -1)

    # Pearson per batch then mean
    eps = 1e-8
    b_mean = b.mean(dim=1, keepdim=True)
    g_mean = g.mean(dim=1, keepdim=True)
    b0 = b - b_mean
    g0 = g - g_mean
    cov = (b0 * g0).mean(dim=1)
    b_std = b0.pow(2).mean(dim=1).sqrt().clamp_min(eps)
    g_std = g0.pow(2).mean(dim=1).sqrt().clamp_min(eps)
    corr = cov / (b_std * g_std)
    return float(corr.mean().item())


def stability_kl(s1: torch.Tensor, s1_aug: torch.Tensor) -> float:
    """
    Average KL divergence KL(s1 || s1_aug) over pixels and batch.
    s1, s1_aug: (B,K,H,W)
    """
    eps = 1e-8
    p = s1.clamp_min(eps)
    q = s1_aug.clamp_min(eps)
    kl = (p * (p.log() - q.log())).sum(dim=1)  # (B,H,W)
    return float(kl.mean().item())


def compute_m1_metrics(
    x: torch.Tensor,
    s1: torch.Tensor,
    boundary: torch.Tensor,
    s1_aug: torch.Tensor,
) -> M1Metrics:
    h_mean, h_std = entropy_stats(s1)
    gini = region_area_gini(s1)
    corr = boundary_gradient_correlation(x, boundary)
    kl = stability_kl(s1, s1_aug)
    return M1Metrics(
        entropy_mean=h_mean,
        entropy_std=h_std,
        region_area_gini=gini,
        boundary_grad_corr=corr,
        stability_kl=kl,
    )

