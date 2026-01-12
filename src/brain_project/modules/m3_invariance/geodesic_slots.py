# FILE: src/brain_project/modules/m3_invariance/geodesic_slots.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
import heapq
import math

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# Data structure
# ============================================================

@dataclass
class GeoSlotsOut:
    masks: torch.Tensor      # (B,S,H,W) soft masks
    labels: torch.Tensor     # (B,H,W) hard labels
    protos: torch.Tensor     # (B,S,F)
    dist: torch.Tensor       # (B,S,H,W) final distance maps used for softmax
    seeds: torch.Tensor      # (B,S,2) y,x


# ============================================================
# Utils
# ============================================================

def _norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def _coords(B: int, H: int, W: int, device, dtype):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,2,H,W)


def build_pixel_features(
    s2: torch.Tensor,                       # (B,K,H,W)
    add_coords: bool = True,
    coord_scale: float = 0.35,
) -> torch.Tensor:
    B, K, H, W = s2.shape
    f = s2
    if add_coords:
        c = _coords(B, H, W, s2.device, s2.dtype) * coord_scale
        f = torch.cat([f, c], dim=1)
    return f  # (B,F,H,W)


def _ensure_dir(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.as_posix(), arr)


def _to_uint8_img01(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)


def _save_gray_png(path: Path, x01: np.ndarray) -> None:
    """
    x01: (H,W) float in [0,1]
    """
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(_to_uint8_img01(x01), mode="L")
    im.save(path.as_posix())


def _save_label_png(path: Path, lab: np.ndarray) -> None:
    """
    lab: (H,W) int
    """
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    # map labels to colors via modulo on a simple palette
    palette = np.array([
        [ 68,  1, 84],
        [ 59, 82,139],
        [ 33,145,140],
        [ 94,201, 97],
        [253,231, 37],
        [255,127, 14],
        [214, 39, 40],
        [148,103,189],
        [140, 86, 75],
        [227,119,194],
    ], dtype=np.uint8)
    H, W = lab.shape
    col = palette[np.clip(lab, 0, 10_000) % len(palette)]
    img = col.reshape(H, W, 3)
    Image.fromarray(img, mode="RGB").save(path.as_posix())


def _save_overlay_seeds(path: Path, base_rgb01: np.ndarray, seeds_yx: np.ndarray) -> None:
    """
    base_rgb01: (H,W,3) float [0,1]
    seeds_yx: (S,2) int (y,x)
    """
    from PIL import Image, ImageDraw, ImageFont
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray((np.clip(base_rgb01, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)

    # try default font; if missing, PIL handles
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    for s, (y, x) in enumerate(seeds_yx.tolist()):
        c = colors[s % len(colors)]
        r = 6
        dr.line([(x - r, y - r), (x + r, y + r)], fill=c, width=2)
        dr.line([(x - r, y + r), (x + r, y - r)], fill=c, width=2)
        dr.text((x + 8, y + 2), f"{s}", fill=c, font=font)

    img.save(path.as_posix())


def _entropy_per_pixel(masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    masks: (B,S,H,W) sum_S=1
    returns: (B,1,H,W) entropy normalized [0,1] approx
    """
    B, S, H, W = masks.shape
    p = masks.clamp_min(eps)
    ent = -(p * torch.log(p)).sum(dim=1, keepdim=True)  # (B,1,H,W)
    ent = ent / math.log(S + 1e-9)
    return ent


# ============================================================
# Seeds
# ============================================================

def pick_seeds_from_s2(
    s2: torch.Tensor,
    num_slots: int,
    min_sep: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """
    Returns: (B,S,2) integer (y,x)
    """
    B, K, H, W = s2.shape
    rng = torch.Generator(device=s2.device)
    rng.manual_seed(seed)

    seeds = []
    for b in range(B):
        chosen = []

        for k in range(min(K, num_slots)):
            idx = torch.argmax(s2[b, k]).item()
            y = idx // W
            x = idx % W
            ok = True
            for (yy, xx) in chosen:
                if (yy - y) ** 2 + (xx - x) ** 2 < min_sep ** 2:
                    ok = False
                    break
            if ok:
                chosen.append((y, x))
            if len(chosen) >= num_slots:
                break

        while len(chosen) < num_slots:
            y = torch.randint(0, H, (1,), generator=rng).item()
            x = torch.randint(0, W, (1,), generator=rng).item()
            chosen.append((y, x))

        seeds.append(chosen)

    return torch.tensor(seeds, dtype=torch.long, device=s2.device)


def pick_perceptual_seeds(
    s2: torch.Tensor,              # (B,K,H,W)
    barrier: torch.Tensor,         # (B,1,H,W)
    num_slots: int,
    min_sep: int = 12,
    seed: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    M3.6b — multi-family perceptual seeds.
    """
    B, K, H, W = s2.shape
    device = s2.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    b = _norm01(barrier[:, :1], eps=eps)  # (B,1,H,W)

    p = s2.clamp_min(0.0)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    pmax = p.max(dim=1, keepdim=True).values  # (B,1,H,W)

    dy = torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])
    dx = torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])
    dy = F.pad(dy, (0, 0, 1, 0))
    dx = F.pad(dx, (1, 0, 0, 0))
    grad = dx + dy
    grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + eps)

    score_f0 = (1.0 - b)                          # interior
    score_f1 = torch.exp(-((b - 0.5) ** 2) / 0.05) # contour band
    score_f2 = pmax                                # strong S2
    score_f3 = (1.0 - grad) * (1.0 - pmax)        # flat/background

    scores = [score_f0, score_f1, score_f2, score_f3]

    base = num_slots // len(scores)
    rem = num_slots % len(scores)
    per_family = [base + (1 if i < rem else 0) for i in range(len(scores))]

    seeds_out = []
    for bb in range(B):
        chosen: list[tuple[int, int]] = []

        for fam_idx, sf in enumerate(scores):
            need = per_family[fam_idx]
            if need <= 0:
                continue

            sflat = sf[bb, 0].flatten()
            idxs = torch.argsort(sflat, descending=True)

            added = 0
            for idx in idxs.tolist():
                y = idx // W
                x = idx % W

                ok = True
                for (yy, xx) in chosen:
                    if (yy - y) ** 2 + (xx - x) ** 2 < (min_sep ** 2):
                        ok = False
                        break
                if not ok:
                    continue

                chosen.append((y, x))
                added += 1
                if added >= need:
                    break

        while len(chosen) < num_slots:
            y = torch.randint(0, H, (1,), generator=rng).item()
            x = torch.randint(0, W, (1,), generator=rng).item()
            chosen.append((y, x))

        seeds_out.append(chosen[:num_slots])

    return torch.tensor(seeds_out, dtype=torch.long, device=device)


# ============================================================
# Dijkstra per slot (S times)
# ============================================================

def _geodesic_single_source_dijkstra(
    step_cost: torch.Tensor,   # (H,W) CPU float
    seed_yx: Tuple[int, int],  # (y,x)
) -> torch.Tensor:
    """
    Returns best(H,W): geodesic distance map from one seed.
    CPU implementation with heapq.
    """
    H, W = step_cost.shape
    inf = 1e9

    best = torch.full((H, W), inf, dtype=torch.float32)
    y0, x0 = int(seed_yx[0]), int(seed_yx[1])
    best[y0, x0] = 0.0

    heap = [(0.0, y0, x0)]
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while heap:
        d, y, x = heapq.heappop(heap)
        if d != float(best[y, x]):
            continue
        for dy, dx in nbrs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                nd = d + float(step_cost[yy, xx])
                if nd < float(best[yy, xx]):
                    best[yy, xx] = nd
                    heapq.heappush(heap, (nd, yy, xx))

    return best


# ============================================================
# Main algorithm (FIX + DEBUG)
# ============================================================

def geodesic_em_slots(
    s2: torch.Tensor,
    barrier: torch.Tensor,
    num_slots: int = 6,
    em_iters: int = 6,
    tau: float = 0.25,
    add_coords: bool = True,
    coord_scale: float = 0.35,
    w_barrier: float = 6.0,
    w_feat: float = 1.0,
    dijkstra_down: int = 1,
    seed_min_sep: int = 12,
    seed_mode: str = "perceptual",
    seeds_yx: torch.Tensor | None = None,
    seed: int = 0,
    eps: float = 1e-6,

    # ---- DEBUG ----
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "geo",
    debug_every: int = 1,           # save every iteration
    debug_save_npy: bool = True,
    debug_save_png: bool = True,
    debug_base_rgb01: Optional[np.ndarray] = None,  # (H,W,3) for overlay seeds
):
    """
    M3.4 + M3.6 (FIXED) + Debug lourd.

    Fix crucial:
      - compute geodesic distances per slot (single-source dijkstra),
        not one shared "best-to-nearest-seed" map.
    """
    assert s2.ndim == 4 and barrier.ndim == 4
    B, K, H, W = s2.shape
    device = s2.device

    debug_dir = _ensure_dir(debug_dir)
    if debug_dir is not None:
        (debug_dir / "meta").mkdir(parents=True, exist_ok=True)
        (debug_dir / "iters").mkdir(parents=True, exist_ok=True)

        # dump a tiny meta file
        meta = {
            "B": B, "K": K, "H": H, "W": W,
            "num_slots": num_slots,
            "em_iters": em_iters,
            "tau": tau,
            "add_coords": add_coords,
            "coord_scale": coord_scale,
            "w_barrier": w_barrier,
            "w_feat": w_feat,
            "dijkstra_down": dijkstra_down,
            "seed_min_sep": seed_min_sep,
            "seed_mode": seed_mode,
            "seed": seed,
        }
        (debug_dir / "meta" / f"{debug_prefix}_meta.txt").write_text(
            "\n".join([f"{k}: {v}" for k, v in meta.items()])
        )

    # ------------------------------------------------
    # Features
    # ------------------------------------------------
    f = build_pixel_features(s2, add_coords=add_coords, coord_scale=coord_scale)  # (B,F,H,W)
    _, Fch, _, _ = f.shape

    b = _norm01(barrier[:, :1], eps=eps)  # (B,1,H,W)

    # ------------------------------------------------
    # Seeds
    # ------------------------------------------------
    if seeds_yx is not None:
        seeds = seeds_yx.to(device)
    else:
        if seed_mode == "perceptual":
            seeds = pick_perceptual_seeds(
                s2, barrier, num_slots=num_slots, min_sep=seed_min_sep, seed=seed, eps=eps
            )
        else:
            seeds = pick_seeds_from_s2(
                s2, num_slots=num_slots, min_sep=seed_min_sep, seed=seed
            )

    # debug save seeds
    if debug_dir is not None:
        seeds_np = seeds.detach().cpu().numpy()
        _save_np(debug_dir / "meta" / f"{debug_prefix}_seeds_yx.npy", seeds_np)
        if debug_base_rgb01 is not None:
            _save_overlay_seeds(
                debug_dir / "meta" / f"{debug_prefix}_seeds_overlay.png",
                debug_base_rgb01,
                seeds_np[0],
            )
        # also save barrier
        if debug_save_png:
            _save_gray_png(
                debug_dir / "meta" / f"{debug_prefix}_barrier.png",
                b[0, 0].detach().cpu().numpy(),
            )

    # ------------------------------------------------
    # Init protos
    # ------------------------------------------------
    protos = torch.zeros((B, num_slots, Fch), device=device)
    for bb in range(B):
        for s in range(num_slots):
            y, x = seeds[bb, s]
            protos[bb, s] = f[bb, :, y, x]

    # ------------------------------------------------
    # Optional downsample
    # ------------------------------------------------
    if dijkstra_down > 1:
        Hd, Wd = H // dijkstra_down, W // dijkstra_down
        f_d = F.interpolate(f, size=(Hd, Wd), mode="bilinear", align_corners=False)
        b_d = F.interpolate(b, size=(Hd, Wd), mode="bilinear", align_corners=False)
    else:
        Hd, Wd = H, W
        f_d, b_d = f, b

    dist_all = torch.zeros((B, num_slots, Hd, Wd), device="cpu", dtype=torch.float32)

    # ============================================================
    # EM LOOP
    # ============================================================
    for it in range(em_iters):
        tau_it = max(tau * (0.85 ** it), 0.05)

        # --- per batch ---
        for bb in range(B):
            step = (1.0 + w_barrier * b_d[bb, 0]).detach().cpu().float()  # (Hd,Wd)

            fd = f_d[bb].detach().cpu().float()          # (F,Hd,Wd)
            fd_hw = fd.view(Fch, -1).t().contiguous()    # (HW,F)
            p = protos[bb].detach().cpu().float()        # (S,F)

            d_feat = ((fd_hw[:, None, :] - p[None, :, :]) ** 2).sum(-1)   # (HW,S)

            sd = seeds[bb].clone()
            sd[:, 0] = (sd[:, 0] // dijkstra_down).clamp(0, Hd - 1)
            sd[:, 1] = (sd[:, 1] // dijkstra_down).clamp(0, Wd - 1)
            sd_cpu = sd.detach().cpu()

            # --- per slot geodesic ---
            for s in range(num_slots):
                best_s = _geodesic_single_source_dijkstra(step, (sd_cpu[s, 0], sd_cpu[s, 1]))  # (Hd,Wd)
                feat_s = _norm01(d_feat[:, s].view(Hd, Wd))                                    # (Hd,Wd)
                dist_all[bb, s] = best_s + (0.35 * w_feat) * feat_s

        # -------- soft assignment --------
        dist_t = dist_all.to(device)  # (B,S,Hd,Wd)
        masks_d = torch.softmax(-dist_t / tau_it, dim=1)

        # -------- upsample --------
        if dijkstra_down > 1:
            masks = F.interpolate(masks_d, size=(H, W), mode="bilinear", align_corners=False)
            dist_up = F.interpolate(dist_t, size=(H, W), mode="bilinear", align_corners=False)
        else:
            masks = masks_d
            dist_up = dist_t

        masks = masks / (masks.sum(dim=1, keepdim=True) + eps)

        # -------- M-step : proto update --------
        for bb in range(B):
            for s in range(num_slots):
                w = masks[bb, s:s+1]  # (1,H,W)
                proto_new = (f[bb] * w).view(Fch, -1).sum(dim=1) / (w.sum() + eps)

                lock = 0.6 if it < (em_iters // 2) else 0.15
                protos[bb, s] = (1 - lock) * protos[bb, s] + lock * proto_new

            # anti-collapse
            P = protos[bb]
            dpp = torch.cdist(P, P)
            if ((dpp < 0.25) & (dpp > 0)).any():
                protos[bb] = P + 0.02 * torch.randn_like(P)

        # ============================================================
        # DEBUG SAVE
        # ============================================================
        if debug_dir is not None and (it % max(debug_every, 1) == 0):
            it_dir = debug_dir / "iters" / f"{debug_prefix}_it{it:02d}"
            it_dir.mkdir(parents=True, exist_ok=True)

            # core tensors
            if debug_save_npy:
                _save_np(it_dir / "masks.npy", masks[0].detach().cpu().numpy())          # (S,H,W)
                _save_np(it_dir / "dist.npy", dist_up[0].detach().cpu().numpy())         # (S,H,W)
                _save_np(it_dir / "protos.npy", protos[0].detach().cpu().numpy())        # (S,F)
                _save_np(it_dir / "labels.npy", masks[0].argmax(dim=0).detach().cpu().numpy())

            # quick PNGs
            if debug_save_png:
                # entropy map (collapse detector)
                ent = _entropy_per_pixel(masks[:1]).detach().cpu().numpy()[0, 0]
                _save_gray_png(it_dir / "entropy.png", ent)

                # labels
                labels = masks[0].argmax(dim=0).detach().cpu().numpy().astype(np.int32)
                _save_label_png(it_dir / "labels.png", labels)

                # each slot mask + dist
                for s in range(min(num_slots, 12)):  # cap to avoid spam
                    ms = masks[0, s].detach().cpu().numpy()
                    ds = dist_up[0, s].detach().cpu().numpy()

                    _save_gray_png(it_dir / f"mask_s{s:02d}.png", _norm01(torch.from_numpy(ms)).numpy())
                    _save_gray_png(it_dir / f"dist_s{s:02d}.png", _norm01(torch.from_numpy(ds)).numpy())

            # numeric diagnostics
            # - occupancy per slot
            occ = masks[0].mean(dim=(1, 2)).detach().cpu().numpy()  # (S,)
            # - proto distances
            pd = torch.cdist(protos[0], protos[0]).detach().cpu().numpy()
            # - min off-diagonal proto dist (collapse score)
            pd2 = pd.copy()
            np.fill_diagonal(pd2, 1e9)
            min_off = float(pd2.min()) if pd2.size else float("nan")

            txt = []
            txt.append(f"it={it} tau_it={tau_it:.4f}")
            txt.append("occupancy mean per slot: " + " ".join([f"{v:.4f}" for v in occ]))
            txt.append(f"min off-diagonal proto dist: {min_off:.4f}")
            (it_dir / "stats.txt").write_text("\n".join(txt))

            # histos (numpy only)
            if debug_save_npy:
                _save_np(it_dir / "occ.npy", occ)
                _save_np(it_dir / "proto_cdist.npy", pd)

    # ------------------------------------------------
    # Final outputs
    # ------------------------------------------------
    masks_full = masks
    dist_full = dist_up
    labels_full = masks_full.argmax(dim=1)

    return GeoSlotsOut(
        masks=masks_full,
        labels=labels_full,
        protos=protos,
        dist=dist_full,
        seeds=seeds,
    )

