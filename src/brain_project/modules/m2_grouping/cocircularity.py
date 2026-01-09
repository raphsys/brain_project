from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn.functional as F


@dataclass
class CoCircularityOut:
    ori: torch.Tensor        # (B,1,H,W) orientation in radians [0,pi)
    cocirc: torch.Tensor     # (B,1,H,W) in [0,1]
    completed: torch.Tensor  # (B,1,H,W) in [0,1]
    edge_norm: torch.Tensor  # (B,1,H,W) normalized edge


def _sobel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def _roll_zeros(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
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
    B = x.shape[0]
    xf = x.view(B, -1)
    lo = xf.quantile(0.02, dim=1, keepdim=True)
    hi = xf.quantile(0.98, dim=1, keepdim=True)
    y = (xf - lo) / (hi - lo + eps)
    return y.clamp(0.0, 1.0).view_as(x)


def _wrap_pi(theta: torch.Tensor) -> torch.Tensor:
    # wrap angle to [0, pi)
    return theta % torch.pi


def _ang_diff_pi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # minimal angular difference for angles modulo pi (orientation, not direction)
    d = torch.abs(a - b)
    return torch.minimum(d, torch.pi - d)


class CoCircularity:
    """
    M2.2 Co-circularity + contour completion (CPU-friendly).
    - Estimate local orientation from gradients of edge map.
    - Compute orientation-consistent affinities to neighbors (8 dirs).
    - Build cocircular score and run few steps of oriented diffusion along tangents.
    """

    def __init__(
        self,
        sigma_theta: float = 0.35,  # ~ 20 degrees
        sigma_dist: float = 1.25,
        iters: int = 8,
        alpha: float = 0.20,
        eps: float = 1e-6,
    ):
        self.sigma_theta = sigma_theta
        self.sigma_dist = sigma_dist
        self.iters = iters
        self.alpha = alpha
        self.eps = eps

        # 8-neighborhood
        self.dirs: List[Tuple[int, int]] = [
            (0, 1),   (-1, 1),  (-1, 0), (-1, -1),
            (0, -1),  (1, -1),  (1, 0),  (1, 1),
        ]
        self.dist = torch.tensor([1.0, 1.4142, 1.0, 1.4142, 1.0, 1.4142, 1.0, 1.4142])

    @torch.no_grad()
    def __call__(self, edge: torch.Tensor) -> CoCircularityOut:
        assert edge.ndim == 4 and edge.shape[1] == 1, "edge must be (B,1,H,W)"
        x = _normalize01_per_image(edge, eps=self.eps)

        # orientation: tangent direction of contours
        gx, gy = _sobel(x)
        # gradient angle gives across-edge direction; tangent is +90 degrees
        theta_g = torch.atan2(gy, gx)                 # [-pi, pi]
        theta_t = _wrap_pi(theta_g + torch.pi / 2.0)  # [0, pi)

        # neighbor affinities based on:
        # - edge strength (both points)
        # - orientation similarity (co-linearity / co-circularity proxy)
        # - distance penalty
        B, _, H, W = x.shape
        cocirc_acc = torch.zeros_like(x)

        # We'll also compute directional weights for diffusion
        w_list = []

        for idx, (dy, dx) in enumerate(self.dirs):
            xn = _roll_zeros(x, dy, dx)
            tn = _roll_zeros(theta_t, dy, dx)

            # orientation compatibility
            dth = _ang_diff_pi(theta_t, tn)
            w_theta = torch.exp(-(dth * dth) / (2.0 * self.sigma_theta * self.sigma_theta))

            # distance penalty
            dist = self.dist[idx].to(x.device, x.dtype)
            w_dist = torch.exp(-(dist * dist) / (2.0 * self.sigma_dist * self.sigma_dist))

            # strength gate: both must be edges-ish
            w_edge = (x * xn).clamp(0.0, 1.0)

            w = (w_theta * w_dist) * w_edge
            w_list.append(w)

            # cocircular score: sum of consistent neighbors
            cocirc_acc = cocirc_acc + w

        # Normalize cocircular score to [0,1]
        cocirc = (cocirc_acc / (cocirc_acc.max().clamp_min(self.eps))).clamp(0.0, 1.0)

        # Contour completion via oriented diffusion:
        # propagate edge strength along orientation-consistent connections.
        u = x.clone()
        for _ in range(self.iters):
            delta = torch.zeros_like(u)
            for (dy, dx), w in zip(self.dirs, w_list):
                un = _roll_zeros(u, dy, dx)
                delta = delta + w * (un - u)
            u = (u + self.alpha * delta).clamp(0.0, 1.0)

        completed = u

        return CoCircularityOut(
            ori=theta_t,
            cocirc=cocirc,
            completed=completed,
            edge_norm=x,
        )

