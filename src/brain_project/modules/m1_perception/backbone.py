from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class BackboneOut:
    feat: torch.Tensor          # (B, C, Hf, Wf)
    stride: int                 # effective stride vs input (approx)
    channels: int               # C


class MobileNetV3SmallBackbone(nn.Module):
    """
    CPU-friendly pretrained visual front-end.
    We use torchvision mobilenet_v3_small pretrained on ImageNet.

    Output is a spatial feature map (B, C, Hf, Wf).
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        # Torchvision API differences across versions:
        # - older: pretrained=True
        # - newer: weights=...
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
        except Exception:
            net = models.mobilenet_v3_small(pretrained=pretrained)

        self.features = net.features  # Sequential
        self.out_channels = 576       # mobilenet_v3_small final feature channels (typical)

        # Freeze by default (evolutionary prior)
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

        # Put batchnorm etc. in eval mode if frozen (stability)
        self._frozen = freeze
        if freeze:
            self.features.eval()

    @torch.no_grad()
    def forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> BackboneOut:
        if self._frozen:
            with torch.no_grad():
                feat = self.features(x)
        else:
            feat = self.features(x)

        # For mobilenet_v3_small, stride is roughly 32 on typical configs.
        # We keep it as metadata; exact stride depends on input size.
        return BackboneOut(feat=feat, stride=32, channels=feat.shape[1])

