from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DepthOut:
    depth: torch.Tensor        # (B,1,H,W) normalized to [0,1]


class MiDaSSmallDepth(torch.nn.Module):
    """
    MiDaS-small via torch.hub (intel-isl/MiDaS).
    CPU-friendly if you keep resolution moderate.
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._frozen = freeze

        # MiDaS uses ImageNet-like normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x: torch.Tensor, work_res: int = 256) -> DepthOut:
        """
        x: (B,3,H,W) in [0,1]
        work_res: internal resolution for MiDaS; keep small for CPU speed.
        """
        B, C, H, W = x.shape

        # resize to work_res (keep aspect by stretching; OK for M1 prior)
        x_in = F.interpolate(x, size=(work_res, work_res), mode="bilinear", align_corners=False)
        x_in = (x_in - self.mean) / self.std

        if self._frozen:
            with torch.no_grad():
                d = self.model(x_in)   # (B, work_res, work_res) or (B,1,*,*)
        else:
            d = self.model(x_in)

        if d.dim() == 3:
            d = d.unsqueeze(1)
        elif d.dim() == 4 and d.shape[1] != 1:
            # some variants output (B,C,H,W); reduce
            d = d.mean(dim=1, keepdim=True)

        # resize back
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)

        # normalize per-image to [0,1] (relative depth)
        d_min = d.amin(dim=(2, 3), keepdim=True)
        d_max = d.amax(dim=(2, 3), keepdim=True)
        d = (d - d_min) / (d_max - d_min + 1e-6)

        return DepthOut(depth=d.clamp(0.0, 1.0))

