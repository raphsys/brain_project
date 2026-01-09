from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class V2AffinityOut:
    w_right: torch.Tensor     # (B,1,H,W-1)
    w_down: torch.Tensor      # (B,1,H-1,W)
    boundary: torch.Tensor    # (B,1,H,W)
    boundary_diffused: torch.Tensor  # (B,1,H,W)


def anisotropic_diffusion(
    u: torch.Tensor,
    w_right: torch.Tensor,
    w_down: torch.Tensor,
    iters: int = 6,
    alpha: float = 0.2,
) -> torch.Tensor:
    """
    Edge-preserving diffusion guided by affinities.
    u: (B,1,H,W)
    """
    B, _, H, W = u.shape
    x = u.clone()

    for _ in range(iters):
        # right / left
        diff_r = torch.zeros_like(x)
        diff_l = torch.zeros_like(x)
        diff_d = torch.zeros_like(x)
        diff_u = torch.zeros_like(x)

        diff_r[:, :, :, :-1] = w_right * (x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_l[:, :, :, 1:]  = w_right * (x[:, :, :, :-1] - x[:, :, :, 1:])

        diff_d[:, :, :-1, :] = w_down * (x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_u[:, :, 1:, :]  = w_down * (x[:, :, :-1, :] - x[:, :, 1:, :])

        x = x + alpha * (diff_r + diff_l + diff_d + diff_u)

    return x


def v2_local_affinity(
    feat: torch.Tensor,
    sigma: float = 0.35,
    diffuse_iters: int = 6,
    diffuse_alpha: float = 0.2,
) -> V2AffinityOut:
    """
    V2 grouping with anisotropic diffusion for stability.
    """
    f = feat
    f = f - f.mean(dim=(2, 3), keepdim=True)
    f = f / (f.std(dim=(2, 3), keepdim=True).clamp_min(1e-6))

    df_right = f[:, :, :, 1:] - f[:, :, :, :-1]
    df_down  = f[:, :, 1:, :] - f[:, :, :-1, :]

    d2_right = (df_right * df_right).mean(dim=1, keepdim=True)
    d2_down  = (df_down  * df_down ).mean(dim=1, keepdim=True)

    w_right = torch.exp(-d2_right / (2.0 * sigma * sigma))
    w_down  = torch.exp(-d2_down  / (2.0 * sigma * sigma))

    B, _, H, Wm1 = w_right.shape
    _, _, Hm1, W = w_down.shape

    boundary = torch.zeros((B, 1, H, W), device=f.device, dtype=f.dtype)

    br = (1.0 - w_right)
    bd = (1.0 - w_down)

    boundary[:, :, :, :-1] += br
    boundary[:, :, :, 1:]  += br
    boundary[:, :, :-1, :] += bd
    boundary[:, :, 1:, :]  += bd

    boundary = (boundary / 4.0).clamp(0.0, 1.0)

    # 🔑 NEW: diffusion
    boundary_diffused = anisotropic_diffusion(
        boundary,
        w_right=w_right,
        w_down=w_down,
        iters=diffuse_iters,
        alpha=diffuse_alpha,
    )

    return V2AffinityOut(
        w_right=w_right,
        w_down=w_down,
        boundary=boundary,
        boundary_diffused=boundary_diffused,
    )

