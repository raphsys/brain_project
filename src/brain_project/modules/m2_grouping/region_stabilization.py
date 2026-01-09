from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RegionStabOut:
    s2: torch.Tensor          # (B,K,H,W) probabilities
    s2_logits: torch.Tensor   # (B,K,H,W) logits after diffusion
    barrier: torch.Tensor     # (B,1,H,W) used barrier map in [0,1]


def _roll_zeros_ch(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Zero-padded shift (no wrap-around).
    x: (B,C,H,W)
    """
    B, C, H, W = x.shape
    y = torch.zeros_like(x)

    y0_src = max(0, -dy); y1_src = min(H, H - dy)
    x0_src = max(0, -dx); x1_src = min(W, W - dx)

    y0_dst = max(0, dy);  y1_dst = min(H, H + dy)
    x0_dst = max(0, dx);  x1_dst = min(W, W + dx)

    if (y1_src > y0_src) and (x1_src > x0_src):
        y[:, :, y0_dst:y1_dst, x0_dst:x1_dst] = x[:, :, y0_src:y1_src, x0_src:x1_src]
    return y


def _normalize01_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B,1,H,W) -> per-image robust normalization to [0,1]
    """
    B = x.shape[0]
    xf = x.view(B, -1)
    lo = xf.quantile(0.02, dim=1, keepdim=True)
    hi = xf.quantile(0.98, dim=1, keepdim=True)
    y = (xf - lo) / (hi - lo + eps)
    return y.clamp(0.0, 1.0).view_as(x)


@torch.no_grad()
def stabilize_regions(
    s1: torch.Tensor,
    barrier_m1: torch.Tensor,
    completed_m2: Optional[torch.Tensor] = None,
    depth: Optional[torch.Tensor] = None,
    iters: int = 10,
    alpha: float = 0.35,
    beta_barrier: float = 6.0,
    beta_depth: float = 2.0,
    eps: float = 1e-6,
) -> RegionStabOut:
    """
    Diffuse logits of S1 under edge barriers to get S2.

    s1: (B,K,H,W) softmax probs
    barrier_m1: (B,1,H,W) boundary map (already diffused) any range
    completed_m2: (B,1,H,W) completed contour map any range (optional)
    depth: (B,1,H,W) normalized depth in [0,1] (optional)

    returns s2 (B,K,H,W).
    """
    assert s1.ndim == 4, "s1 must be (B,K,H,W)"
    assert barrier_m1.ndim == 4 and barrier_m1.shape[1] == 1

    B, K, H, W = s1.shape

    # build barrier = combine M1 boundary + M2 completed contours (strong barriers)
    b1 = _normalize01_per_image(barrier_m1, eps=eps)

    if completed_m2 is not None:
        assert completed_m2.ndim == 4 and completed_m2.shape[1] == 1
        c2 = _normalize01_per_image(completed_m2, eps=eps)
        barrier = torch.maximum(b1, c2)   # strongest wins
    else:
        barrier = b1

    # Convert probs to logits for stable diffusion in log-space-ish
    s1 = s1.clamp(eps, 1.0)
    logits = torch.log(s1)  # (B,K,H,W)

    # Depth gate
    if depth is not None:
        assert depth.ndim == 4 and depth.shape[1] == 1
        d = depth.clamp(0.0, 1.0)
    else:
        d = None

    # 4-neighborhood (cheaper and stable)
    dirs: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))

    # Precompute barrier weights to neighbors: w = exp(-beta * barrier_between)
    # barrier_between approximated by max(barrier[p], barrier[q])
    def neighbor_weight(dy: int, dx: int) -> torch.Tensor:
        b_n = _roll_zeros_ch(barrier, dy, dx)
        b_between = torch.maximum(barrier, b_n)  # (B,1,H,W)
        w = torch.exp(-beta_barrier * b_between).clamp(0.0, 1.0)  # (B,1,H,W)

        if d is not None:
            d_n = _roll_zeros_ch(d, dy, dx)
            dd = torch.abs(d - d_n)
            w_d = torch.exp(-beta_depth * dd).clamp(0.0, 1.0)
            w = w * w_d

        return w  # (B,1,H,W)

    weights = [neighbor_weight(dy, dx) for dy, dx in dirs]

    # Iterative anisotropic diffusion on logits
    for _ in range(iters):
        delta = torch.zeros_like(logits)
        wsum = torch.zeros((B, 1, H, W), device=logits.device, dtype=logits.dtype)

        for (dy, dx), w in zip(dirs, weights):
            ln = _roll_zeros_ch(logits, dy, dx)  # (B,K,H,W)
            delta = delta + (w * (ln - logits))
            wsum = wsum + w

        # Normalize by total weight to prevent oversmoothing variation across pixels
        logits = logits + alpha * (delta / (wsum + eps))

    # Back to probabilities
    s2 = torch.softmax(logits, dim=1)

    return RegionStabOut(
        s2=s2,
        s2_logits=logits,
        barrier=barrier,
    )

