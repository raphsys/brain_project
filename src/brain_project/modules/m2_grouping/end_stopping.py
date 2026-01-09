from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EndStoppingOut:
    end_map: torch.Tensor      # (B,1,H,W) in [0,1]
    ori_bins: torch.Tensor     # (B,1,H,W) int64 in [0..n_bins-1]
    edge_norm: torch.Tensor    # (B,1,H,W) normalized edge strength


def _sobel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B,1,H,W)
    returns gx, gy (B,1,H,W)
    """
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
    """
    torch.roll but with zero padding instead of wrap-around.
    x: (B,1,H,W)
    """
    B, C, H, W = x.shape
    y = torch.zeros_like(x)

    y0_src = max(0, -dy)
    y1_src = min(H, H - dy)   # exclusive
    x0_src = max(0, -dx)
    x1_src = min(W, W - dx)

    y0_dst = max(0, dy)
    y1_dst = min(H, H + dy)
    x0_dst = max(0, dx)
    x1_dst = min(W, W + dx)

    if (y1_src > y0_src) and (x1_src > x0_src):
        y[:, :, y0_dst:y1_dst, x0_dst:x1_dst] = x[:, :, y0_src:y1_src, x0_src:x1_src]
    return y


def _quantize_orientation(gx: torch.Tensor, gy: torch.Tensor, n_bins: int = 8) -> torch.Tensor:
    """
    Quantize angle of gradient into n_bins.
    gx, gy: (B,1,H,W)
    returns bins: int64 (B,1,H,W)
    """
    # angle in [-pi, pi]
    ang = torch.atan2(gy, gx)  # (B,1,H,W)
    # map to [0, 2pi)
    ang = ang % (2.0 * torch.pi)
    # bin
    bins = torch.floor(ang / (2.0 * torch.pi / n_bins)).to(torch.int64)
    bins = torch.clamp(bins, 0, n_bins - 1)
    return bins


class EndStopping(nn.Module):
    """
    M2.1 End-stopping operator:
    - input: edge map (V2 boundary diffused) (B,1,H,W)
    - output: end_map (B,1,H,W) where high => likely end of a contour segment.

    CPU-friendly: uses only conv2d + shifts.
    """

    def __init__(
        self,
        n_bins: int = 8,
        radius: int = 6,
        edge_smooth: int = 0,   # optional avgpool smoothing on edge map (0 disables)
        eps: float = 1e-6,
    ):
        super().__init__()
        assert n_bins in (4, 8, 16)
        self.n_bins = n_bins
        self.radius = radius
        self.edge_smooth = edge_smooth
        self.eps = eps

        # Discrete direction offsets for bins (approx. unit steps)
        # These directions represent the "tangent" direction along the contour.
        # For simplicity, we use 8-neighborhood directions.
        # bins: 0..7 correspond to angles 0,45,90,...
        self._dirs_8: List[Tuple[int, int]] = [
            (0, 1),    # 0: right
            (-1, 1),   # 1: up-right
            (-1, 0),   # 2: up
            (-1, -1),  # 3: up-left
            (0, -1),   # 4: left
            (1, -1),   # 5: down-left
            (1, 0),    # 6: down
            (1, 1),    # 7: down-right
        ]

    @torch.no_grad()
    def forward(self, edge: torch.Tensor) -> EndStoppingOut:
        """
        edge: (B,1,H,W) raw boundary strength (any range)
        """
        assert edge.ndim == 4 and edge.shape[1] == 1, "edge must be (B,1,H,W)"
        x = edge

        # Normalize edge to [0,1] per-image (robust for real images)
        B = x.shape[0]
        x_flat = x.view(B, -1)
        lo = x_flat.quantile(0.02, dim=1, keepdim=True)
        hi = x_flat.quantile(0.98, dim=1, keepdim=True)
        x_norm = (x_flat - lo) / (hi - lo + self.eps)
        x_norm = x_norm.clamp(0.0, 1.0).view_as(x)

        if self.edge_smooth and self.edge_smooth > 0:
            k = self.edge_smooth
            x_norm = F.avg_pool2d(x_norm, kernel_size=k, stride=1, padding=k // 2)

        # Orientation from Sobel of edge map (simple and good enough for M2.1)
        gx, gy = _sobel(x_norm)
        ori_bins = _quantize_orientation(gx, gy, n_bins=self.n_bins)  # (B,1,H,W)

        # We want direction ALONG the contour, not across it.
        # Gradient points across edges; tangent is rotated by +90°.
        # So we shift bins by n_bins/4 (i.e., +90 degrees).
        rot = self.n_bins // 4
        tan_bins = (ori_bins + rot) % self.n_bins  # (B,1,H,W)

        # Forward/backward continuation strength along tangent direction
        # For each pixel, look ahead/behind up to radius and take max edge strength.
        fwd = torch.zeros_like(x_norm)
        bwd = torch.zeros_like(x_norm)

        # For n_bins != 8, we still approximate using 8 directions by mapping.
        # If n_bins=4: map to 0,2,4,6
        # If n_bins=16: still mapped to nearest among 8 directions (coarser).
        def bin_to_dir(bin_id: int) -> Tuple[int, int]:
            if self.n_bins == 8:
                return self._dirs_8[bin_id]
            if self.n_bins == 4:
                return self._dirs_8[(bin_id * 2) % 8]
            # n_bins == 16: map to 8 by /2
            return self._dirs_8[(bin_id // 2) % 8]

        # Precompute shifted edge maps per direction and distance to avoid repeated work
        # But we need per-pixel direction choice. We'll build per-direction stacks.
        dir_shifts = []
        for d in range(8):
            dy, dx = self._dirs_8[d]
            # stack distances 1..radius (max pooling over distances)
            shifted_max_f = torch.zeros_like(x_norm)
            shifted_max_b = torch.zeros_like(x_norm)
            for r in range(1, self.radius + 1):
                shifted_max_f = torch.maximum(shifted_max_f, _roll_zeros(x_norm, dy * r, dx * r))
                shifted_max_b = torch.maximum(shifted_max_b, _roll_zeros(x_norm, -dy * r, -dx * r))
            dir_shifts.append((shifted_max_f, shifted_max_b))

        # Select fwd/bwd per pixel based on tan_bins
        # Build masks per dir
        if self.n_bins == 8:
            bins8 = tan_bins
        elif self.n_bins == 4:
            bins8 = (tan_bins * 2) % 8
        else:  # 16
            bins8 = (tan_bins // 2) % 8

        for d in range(8):
            mask = (bins8 == d).to(x_norm.dtype)  # (B,1,H,W)
            f_d, b_d = dir_shifts[d]
            fwd = fwd + mask * f_d
            bwd = bwd + mask * b_d

        # Endness: edge is strong, but continuity is asymmetric.
        # If both sides continue strongly => middle of a long segment => endness low.
        # If one side weak and other strong => endness high.
        min_fb = torch.minimum(fwd, bwd)
        max_fb = torch.maximum(fwd, bwd)

        # ratio close to 1 => symmetric continuation; ratio small => one-sided => likely end
        ratio = min_fb / (max_fb + self.eps)

        # end_map emphasizes strong edge points with low symmetry
        end_map = x_norm * (1.0 - ratio)
        end_map = end_map.clamp(0.0, 1.0)

        return EndStoppingOut(
            end_map=end_map,
            ori_bins=bins8.to(torch.int64),
            edge_norm=x_norm,
        )

