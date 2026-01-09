from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class V1GaborOut:
    energy: torch.Tensor          # (B, S*O, H, W) energy maps
    energy_sum: torch.Tensor      # (B, 1, H, W)   summed energy
    ori_map: torch.Tensor         # (B, 1, H, W)   dominant orientation index [0..O-1]


def _make_gabor_kernel(
    ksize: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float,
    psi: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (real, imag) gabor kernels of shape (ksize, ksize)."""
    assert ksize % 2 == 1, "ksize should be odd"
    half = ksize // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        indexing="ij",
    )

    # rotation
    x_theta = xs * math.cos(theta) + ys * math.sin(theta)
    y_theta = -xs * math.sin(theta) + ys * math.cos(theta)

    gauss = torch.exp(-(x_theta**2 + (gamma**2) * (y_theta**2)) / (2.0 * sigma**2))
    phase = (2.0 * math.pi * x_theta / lambd) + psi
    real = gauss * torch.cos(phase)
    imag = gauss * torch.sin(phase)

    # zero-mean (helps)
    real = real - real.mean()
    imag = imag - imag.mean()

    # normalize energy
    real = real / (real.norm() + 1e-8)
    imag = imag / (imag.norm() + 1e-8)

    return real, imag


def v1_gabor_energy(
    x: torch.Tensor,
    scales: List[Tuple[int, float, float]] = None,
    orientations: int = 8,
    gamma: float = 0.5,
) -> V1GaborOut:
    """
    x: (B,3,H,W) in [0,1]
    Output energy maps: (B, S*O, H, W) where S=len(scales), O=orientations.

    scales: list of (ksize, sigma, lambd)
    """
    if scales is None:
        # (ksize, sigma, wavelength). Small set for CPU.
        scales = [
            (15, 2.5, 6.0),
            (21, 3.5, 9.0),
        ]

    device = x.device
    dtype = x.dtype

    # grayscale luminance (rough V1 input)
    # (B,1,H,W)
    lum = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).clamp(0, 1)

    kernels_real = []
    kernels_imag = []

    for (ksize, sigma, lambd) in scales:
        for o in range(orientations):
            theta = (math.pi * o) / orientations
            real, imag = _make_gabor_kernel(
                ksize=ksize,
                sigma=sigma,
                theta=theta,
                lambd=lambd,
                gamma=gamma,
                psi=0.0,
                device=device,
                dtype=dtype,
            )
            kernels_real.append(real)
            kernels_imag.append(imag)

    # Stack into conv weights: (F,1,ks,ks)
    # But kernels have varying ksize if scales differ -> we pad to max.
    max_k = max(k.shape[0] for k in kernels_real)
    def pad_to(k: torch.Tensor, K: int) -> torch.Tensor:
        p = (K - k.shape[0]) // 2
        if p == 0:
            return k
        return F.pad(k, (p, p, p, p), mode="constant", value=0.0)

    w_r = torch.stack([pad_to(k, max_k) for k in kernels_real], dim=0).unsqueeze(1)
    w_i = torch.stack([pad_to(k, max_k) for k in kernels_imag], dim=0).unsqueeze(1)

    # Convolve
    pad = max_k // 2
    resp_r = F.conv2d(lum, w_r, padding=pad)  # (B, F, H, W)
    resp_i = F.conv2d(lum, w_i, padding=pad)

    energy = torch.sqrt(resp_r * resp_r + resp_i * resp_i + 1e-8)  # complex-cell energy

    # Summed energy across filters
    energy_sum = energy.mean(dim=1, keepdim=True)

    # Dominant orientation map (ignore scale): reshape (B, S, O, H, W) -> sum over S
    S = len(scales)
    O = orientations
    e_so = energy.view(energy.shape[0], S, O, energy.shape[2], energy.shape[3]).mean(dim=1)  # (B,O,H,W)
    ori_idx = torch.argmax(e_so, dim=1, keepdim=True).to(torch.float32)  # store as float for easy viz

    return V1GaborOut(energy=energy, energy_sum=energy_sum, ori_map=ori_idx)

