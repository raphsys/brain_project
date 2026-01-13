# FILE: src/brain_project/modules/m3_invariance/m3_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from brain_project.modules.m3_invariance.slots_kmeans import soft_kmeans_slots_barrier
from brain_project.modules.m3_invariance.geodesic_slots import geodesic_em_slots
from brain_project.modules.m4_grouping.m4_spatial import m4_spatial_grouping


# ---------------------------
# Basic utilities
# ---------------------------

def gradient_barrier(x01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x01: (B,1,H,W) in [0,1]
    """
    dy = torch.abs(x01[:, :, 1:, :] - x01[:, :, :-1, :])
    dx = torch.abs(x01[:, :, :, 1:] - x01[:, :, :, :-1])

    dy = F.pad(dy, (0, 0, 1, 0))
    dx = F.pad(dx, (1, 0, 0, 0))

    g = dx + dy
    g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g


# ---------------------------
# M3.5 - Lateral inhibition
# ---------------------------

def _gaussian_kernel2d(ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
    assert ksize % 2 == 1, "ksize must be odd"
    ax = torch.arange(ksize, device=device) - (ksize // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def _blur_per_channel(x: torch.Tensor, ksize: int = 11, sigma: float = 2.5) -> torch.Tensor:
    B, K, H, W = x.shape
    kernel = _gaussian_kernel2d(ksize, sigma, x.device).view(1, 1, ksize, ksize)
    weight = kernel.repeat(K, 1, 1, 1)  # (K,1,ks,ks)
    return F.conv2d(x, weight, padding=ksize // 2, groups=K)


def lateral_inhibition(
    masks: torch.Tensor,
    iters: int = 12,
    alpha: float = 1.4,
    beta: float = 0.10,
    blur_ksize: int = 11,
    blur_sigma: float = 2.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    masks: (B,S,H,W) non-negative, sum_S ~ 1
    """
    x = masks.clamp_min(0.0)
    x = x / (x.sum(dim=1, keepdim=True) + eps)

    for _ in range(iters):
        x_blur = _blur_per_channel(x, ksize=blur_ksize, sigma=blur_sigma)
        comp = x_blur.sum(dim=1, keepdim=True) - x_blur
        x = x - alpha * comp + beta * x_blur
        x = F.relu(x)
        x = x / (x.sum(dim=1, keepdim=True) + eps)

    return x


def labels_from_masks(masks: torch.Tensor) -> torch.Tensor:
    return masks.argmax(dim=1)


# ---------------------------
# S2 low-level (bio) : enriched
# ---------------------------

def build_s2_enriched(
    x: torch.Tensor,         # (B,3,H,W) in [0,1]
    depth: torch.Tensor,     # (B,1,H,W) in [0,1]
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    S2_low : luminance + texture + rgb + depth
    """
    img_gray = x.mean(dim=1, keepdim=True)

    gx = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    gy = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])
    gx = F.pad(gx, (1, 0, 0, 0))
    gy = F.pad(gy, (0, 0, 1, 0))
    texture = gx + gy

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    s2 = torch.cat([img_gray, texture, r, g, b, depth], dim=1)

    # contraste local
    s2 = s2 - s2.mean(dim=(2, 3), keepdim=True)
    s2 = s2 / (s2.std(dim=(2, 3), keepdim=True) + eps)

    # firing rates
    s2 = F.relu(s2)

    # mixture-like normalization
    s2 = s2 / (s2.sum(dim=1, keepdim=True) + eps)
    return s2


# ---------------------------
# S2 semantic (weak) : DINOv2
# ---------------------------

def build_s2_from_dinov2(
    x_rgb01: torch.Tensor,          # (B,3,H,W) in [0,1]
    out_size: int = 256,
    num_channels: int = 24,
    model_name: str = "dinov2_vits14",
    seed: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns: (B,num_channels,out_size,out_size) mixture-like
    """
    device = x_rgb01.device
    B, _, H, W = x_rgb01.shape

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.eval().to(device)

    patch = 14
    try:
        ps = model.patch_embed.patch_size
        patch = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
    except Exception:
        pass

    H2 = max((H // patch) * patch, patch)
    W2 = max((W // patch) * patch, patch)

    xin = F.interpolate(x_rgb01, size=(H2, W2), mode="bilinear", align_corners=False)
    xin = (xin - mean) / std

    feats = model.forward_features(xin)
    if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
        pt = feats["x_norm_patchtokens"]
    elif isinstance(feats, dict) and "x_patchtokens" in feats:
        pt = feats["x_patchtokens"]
    else:
        pt = feats

    if pt.ndim != 3:
        raise RuntimeError(f"DINOv2 patch tokens expected (B,N,C), got {tuple(pt.shape)}")

    h = H2 // patch
    w = W2 // patch
    _, N, C = pt.shape

    if N != h * w:
        hw = int(np.sqrt(N))
        if hw * hw == N:
            h = w = hw
        elif N % h == 0:
            w = N // h
        else:
            raise RuntimeError(f"Cannot reshape patch tokens: N={N}, expected {h*w}")

    fmap = pt.transpose(1, 2).contiguous().view(B, C, h, w)
    fmap = fmap / (fmap.norm(dim=1, keepdim=True) + eps)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    Wproj = torch.randn((num_channels, C), generator=g, device=device) / np.sqrt(C)
    proj = torch.einsum("bchw,kc->bkhw", fmap, Wproj)

    proj = F.relu(proj)
    proj = F.interpolate(proj, size=(out_size, out_size), mode="bilinear", align_corners=False)

    s2 = torch.softmax(proj, dim=1)
    return s2


# ---------------------------
# M3 pipeline output (+ M4)
# ---------------------------

@dataclass
class M3Out:
    s2: torch.Tensor
    s2_low: torch.Tensor
    s2_dino: Optional[torch.Tensor]
    barrier: torch.Tensor
    km_masks: torch.Tensor
    geo_masks: torch.Tensor
    geo_labels: torch.Tensor
    seeds: torch.Tensor
    m35_masks: torch.Tensor
    m35_labels: torch.Tensor
    protos: torch.Tensor

    # M4
    m4_labels: Optional[torch.Tensor]
    m4_num_regions: int


def run_m3(
    x_rgb01: torch.Tensor,
    depth01: torch.Tensor,
    num_slots: int = 6,
    use_dino: bool = True,
    dino_channels: int = 24,
    s2_low_channels_expected: int = 6,  # (gray, texture, r,g,b,depth)
    geo_em_iters: int = 6,
    geo_tau: float = 0.20,
    w_barrier: float = 6.0,
    w_feat: float = 1.0,
    seed_min_sep: int = 14,
    seed_mode: str = "perceptual",
    seed: int = 0,
    inhibit: bool = True,
    inhibit_iters: int = 14,
    inhibit_alpha: float = 1.6,
    inhibit_beta: float = 0.12,
    inhibit_blur_ksize: int = 11,
    inhibit_blur_sigma: float = 2.2,
    # M4
    run_m4: bool = True,
    m4_lam_sem: float = 0.45,
    m4_lam_per: float = 0.45,
    m4_lam_cent: float = 0.10,
    m4_tau: float = 0.45,
    eps: float = 1e-6,
) -> M3Out:
    """
    M3 complet : S2 fusion -> barrier -> kmeans -> geodesic EM -> inhibition -> (option) M4 spatial grouping
    """

    # 1) S2_low
    s2_low = build_s2_enriched(x_rgb01, depth01, eps=eps)
    if s2_low.shape[1] != s2_low_channels_expected:
        pass

    # 2) S2_dino
    s2_dino = None
    if use_dino:
        s2_dino = build_s2_from_dinov2(
            x_rgb01,
            out_size=x_rgb01.shape[-1],
            num_channels=dino_channels,
            model_name="dinov2_vits14",
            seed=seed,
            eps=eps,
        )

    # 3) fusion
    if s2_dino is not None:
        s2 = torch.cat([s2_low, s2_dino], dim=1)
    else:
        s2 = s2_low

    s2 = s2.clamp_min(0.0)
    s2 = s2 / (s2.sum(dim=1, keepdim=True) + eps)

    # 4) barrier
    barrier = gradient_barrier(depth01, eps=eps)

    # 5) baseline soft kmeans (debug)
    out_km = soft_kmeans_slots_barrier(
        s2,
        barrier=barrier,
        num_slots=num_slots,
        iters=15,
        tau=0.25,
        seed=seed,
    )
    km_masks = out_km.masks

    # 6) geodesic EM
    out_geo = geodesic_em_slots(
        s2,
        barrier=barrier,
        num_slots=num_slots,
        em_iters=geo_em_iters,
        tau=geo_tau,
        w_barrier=w_barrier,
        w_feat=w_feat,
        dijkstra_down=1,
        seed_min_sep=seed_min_sep,
        seed_mode=seed_mode,
        seed=seed,
    )

    geo_masks = out_geo.masks
    geo_labels = out_geo.labels
    seeds = out_geo.seeds
    protos = out_geo.protos

    # 7) inhibition
    if inhibit:
        m35_masks = lateral_inhibition(
            geo_masks,
            iters=inhibit_iters,
            alpha=inhibit_alpha,
            beta=inhibit_beta,
            blur_ksize=inhibit_blur_ksize,
            blur_sigma=inhibit_blur_sigma,
            eps=eps,
        )
    else:
        m35_masks = geo_masks

    m35_labels = labels_from_masks(m35_masks)  # (B,H,W)

    # 8) M4 spatial grouping (adjacent-only)
    m4_labels = None
    m4_num_regions = 0
    if run_m4:
        # semantic feature map: use s2_dino if present, else zeros
        if s2_dino is not None:
            sem_map = s2_dino
        else:
            sem_map = torch.zeros((x_rgb01.shape[0], 1, x_rgb01.shape[-2], x_rgb01.shape[-1]),
                                  device=x_rgb01.device, dtype=x_rgb01.dtype)

        # perceptual feature map: use s2_low
        per_map = s2_low

        m4 = m4_spatial_grouping(
            labels_m3=m35_labels,
            sem=sem_map,
            per=per_map,
            lam_sem=m4_lam_sem,
            lam_per=m4_lam_per,
            lam_cent=m4_lam_cent,
            tau=m4_tau,
            eps=eps,
        )
        m4_labels = m4.labels
        m4_num_regions = m4.num_regions

    return M3Out(
        s2=s2,
        s2_low=s2_low,
        s2_dino=s2_dino,
        barrier=barrier,
        km_masks=km_masks,
        geo_masks=geo_masks,
        geo_labels=geo_labels,
        seeds=seeds,
        m35_masks=m35_masks,
        m35_labels=m35_labels,
        protos=protos,
        m4_labels=m4_labels,
        m4_num_regions=m4_num_regions,
    )

