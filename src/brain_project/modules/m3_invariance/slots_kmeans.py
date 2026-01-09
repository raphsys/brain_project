from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SlotsOut:
    masks: torch.Tensor     # (B,S,H,W)
    protos: torch.Tensor    # (B,S,F)
    recon: torch.Tensor     # (B,F,H,W)
    weights: torch.Tensor   # (B,HW,S) soft assignments


def _coords(B: int, H: int, W: int, device, dtype):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,2,H,W)


def build_pixel_features(
    s2: torch.Tensor,
    add_coords: bool = True,
    coord_scale: float = 0.35,
) -> torch.Tensor:
    """
    s2: (B,K,H,W) probs
    returns: f (B,F,H,W) where F = K (+2 coords)
    """
    B, K, H, W = s2.shape
    f = s2
    if add_coords:
        c = _coords(B, H, W, s2.device, s2.dtype) * coord_scale
        f = torch.cat([f, c], dim=1)
    return f


def init_protos_from_pixels(
    f: torch.Tensor,
    S: int,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    B, Fch, H, W = f.shape
    HW = H * W
    fp = f.view(B, Fch, HW).transpose(1, 2)  # (B,HW,F)
    idx = torch.randint(0, HW, (B, S), device=f.device)
    protos = torch.gather(fp, 1, idx.unsqueeze(-1).expand(B, S, Fch))  # (B,S,F)
    return protos


def _norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def edge_aware_diffuse(
    masks: torch.Tensor,
    barrier: torch.Tensor,
    steps: int = 6,
    alpha: float = 0.25,
    beta: float = 12.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    masks:   (B,S,H,W) in [0,1]
    barrier: (B,1,H,W) higher = stronger edge barrier
    Diffusion that smooths WITHIN regions and stops across strong edges.
    """
    # conductance: low at edges, high in smooth areas
    b = _norm01(barrier)
    c = torch.exp(-beta * b).clamp(0.0, 1.0)  # (B,1,H,W)

    m = masks
    for _ in range(steps):
        # 4-neighborhood differences
        up    = F.pad(m[:, :, :-1, :], (0, 0, 1, 0))
        down  = F.pad(m[:, :, 1:,  :], (0, 0, 0, 1))
        left  = F.pad(m[:, :, :, :-1], (1, 0, 0, 0))
        right = F.pad(m[:, :, :, 1: ], (0, 1, 0, 0))

        # conductance between pixels (use product to be conservative)
        cup    = F.pad(c[:, :, :-1, :], (0, 0, 1, 0)) * c
        cdown  = F.pad(c[:, :, 1:,  :], (0, 0, 0, 1)) * c
        cleft  = F.pad(c[:, :, :, :-1], (1, 0, 0, 0)) * c
        cright = F.pad(c[:, :, :, 1: ], (0, 1, 0, 0)) * c

        # diffusion update (discrete anisotropic Laplacian)
        lap = (
            cup   * (up   - m) +
            cdown * (down - m) +
            cleft * (left - m) +
            cright* (right- m)
        )

        m = (m + alpha * lap).clamp(0.0, 1.0)

        # renormalize across slots (so each pixel distributes mass over slots)
        m = m / (m.sum(dim=1, keepdim=True) + eps)

    return m


def soft_kmeans_slots_barrier(
    s2: torch.Tensor,
    barrier: Optional[torch.Tensor] = None,  # (B,1,H,W)
    num_slots: int = 6,
    iters: int = 10,
    tau: float = 0.25,
    add_coords: bool = True,
    coord_scale: float = 0.35,
    seed: int = 0,
    # M3.3 params
    diffuse_steps: int = 6,
    diffuse_alpha: float = 0.25,
    diffuse_beta: float = 12.0,
    eps: float = 1e-6,
) -> SlotsOut:
    """
    M3.3: Soft k-means + edge-aware diffusion using a barrier map.
    """
    assert s2.ndim == 4
    B, K, H, W = s2.shape
    device = s2.device

    f = build_pixel_features(s2, add_coords=add_coords, coord_scale=coord_scale)  # (B,F,H,W)
    B, Fch, H, W = f.shape
    HW = H * W
    fp = f.view(B, Fch, HW).transpose(1, 2).contiguous()  # (B,HW,F)

    protos = init_protos_from_pixels(f, num_slots, seed=seed)  # (B,S,F)

    if barrier is None:
        barrier = torch.zeros((B, 1, H, W), device=device, dtype=s2.dtype)
    else:
        if barrier.ndim == 3:
            barrier = barrier.unsqueeze(1)
        barrier = barrier.to(device=device, dtype=s2.dtype)

    for _ in range(iters):
        # distances (B,HW,S)
        dist = ((fp.unsqueeze(2) - protos.unsqueeze(1)) ** 2).sum(dim=-1)

        w = torch.softmax(-dist / max(tau, eps), dim=-1)  # (B,HW,S)
        masks = w.transpose(1, 2).view(B, num_slots, H, W)

        # M3.3 key: barrier-constrained diffusion of masks
        masks = edge_aware_diffuse(
            masks, barrier,
            steps=diffuse_steps,
            alpha=diffuse_alpha,
            beta=diffuse_beta,
            eps=eps,
        )

        # back to weights
        w = masks.view(B, num_slots, HW).transpose(1, 2).contiguous()  # (B,HW,S)

        # update protos
        denom = w.sum(dim=1, keepdim=True).transpose(1, 2)  # (B,S,1)
        num = torch.bmm(w.transpose(1, 2), fp)              # (B,S,F)
        protos = num / (denom + eps)

    # final
    dist = ((fp.unsqueeze(2) - protos.unsqueeze(1)) ** 2).sum(dim=-1)
    w = torch.softmax(-dist / max(tau, eps), dim=-1)
    masks = w.transpose(1, 2).view(B, num_slots, H, W)

    masks = edge_aware_diffuse(
        masks, barrier,
        steps=diffuse_steps,
        alpha=diffuse_alpha,
        beta=diffuse_beta,
        eps=eps,
    )
    w = masks.view(B, num_slots, HW).transpose(1, 2).contiguous()

    recon_fp = torch.bmm(w, protos)  # (B,HW,F)
    recon = recon_fp.transpose(1, 2).view(B, Fch, H, W)

    return SlotsOut(masks=masks, protos=protos, recon=recon, weights=w)

