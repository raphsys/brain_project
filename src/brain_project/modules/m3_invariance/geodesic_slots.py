from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import heapq

import torch
import torch.nn.functional as F


@dataclass
class GeoSlotsOut:
    masks: torch.Tensor      # (B,S,H,W) soft masks
    labels: torch.Tensor     # (B,H,W) hard labels
    protos: torch.Tensor     # (B,S,F)
    dist: torch.Tensor       # (B,S,H,W) geodesic distances


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


def pick_seeds_from_s2(
    s2: torch.Tensor,
    num_slots: int,
    min_sep: int = 10,
) -> torch.Tensor:
    """
    Pick seeds as argmax peaks from different channels, then fall back to random.
    Returns: (B,S,2) integer (y,x)
    """
    B, K, H, W = s2.shape
    seeds = []
    for b in range(B):
        chosen = []
        # try per-channel peaks
        for k in range(min(K, num_slots)):
            idx = torch.argmax(s2[b, k]).item()
            y = idx // W
            x = idx % W
            ok = True
            for (yy, xx) in chosen:
                if (yy - y) * (yy - y) + (xx - x) * (xx - x) < (min_sep * min_sep):
                    ok = False
                    break
            if ok:
                chosen.append((y, x))
            if len(chosen) >= num_slots:
                break

        # fill remaining randomly
        while len(chosen) < num_slots:
            y = torch.randint(0, H, (1,)).item()
            x = torch.randint(0, W, (1,)).item()
            chosen.append((y, x))

        seeds.append(chosen)

    return torch.tensor(seeds, dtype=torch.long, device=s2.device)  # (B,S,2)


def _geodesic_multisource_dijkstra(
    step_cost: torch.Tensor,   # (H,W) >= 0
    seeds_yx: torch.Tensor,    # (S,2) long
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-source Dijkstra on 4-neighborhood.
    Returns:
      dist: (S,H,W) distances from each seed (computed efficiently as label-distance by running one multi-label Dijkstra)
      labels: (H,W) winner label (argmin)
    Implementation detail:
      We compute best distance + label in one pass (multi-label), then reconstruct per-slot dist approximately:
      - exact per-slot dist would require S runs.
      - Here, for slot-soft masks, we mainly need *relative* distances. We'll also output a "dist_to_winner" map and
        a "dist_per_slot" coarse approximation using winner distance + feature distance later.
    """
    H, W = step_cost.shape
    inf = 1e9

    best = torch.full((H, W), inf, dtype=torch.float32)
    lab = torch.full((H, W), -1, dtype=torch.int64)

    # heap entries: (d, y, x, label)
    heap = []
    for s, (y, x) in enumerate(seeds_yx.tolist()):
        best[y, x] = 0.0
        lab[y, x] = s
        heapq.heappush(heap, (0.0, y, x, s))

    # neighbors
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while heap:
        d, y, x, s = heapq.heappop(heap)
        if d != float(best[y, x]) or s != int(lab[y, x]):
            continue
        for dy, dx in nbrs:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                continue
            nd = d + float(step_cost[yy, xx])
            # relax if better
            if nd < float(best[yy, xx]):
                best[yy, xx] = nd
                lab[yy, xx] = s
                heapq.heappush(heap, (nd, yy, xx, s))

    return best, lab


def geodesic_em_slots(
    s2: torch.Tensor,                    # (B,K,H,W)
    barrier: torch.Tensor,               # (B,1,H,W) higher=more edge
    num_slots: int = 6,
    em_iters: int = 5,
    tau: float = 0.25,                   # softness of slot masks from distances
    add_coords: bool = True,
    coord_scale: float = 0.35,
    # costs
    w_barrier: float = 3.0,              # barrier contribution to step cost
    w_feat: float = 1.0,                 # feature mismatch contribution
    # speed knobs
    dijkstra_down: int = 1,              # 1=full res, 2=half res, etc.
    seed_min_sep: int = 10,
    eps: float = 1e-6,
) -> GeoSlotsOut:
    """
    M3.4: Geodesic slots = region growth constrained by barrier (M2) + feature consistency (S2).
    CPU-friendly on ~128x128; can downsample for speed.
    """
    assert s2.ndim == 4 and barrier.ndim == 4
    B, K, H, W = s2.shape
    device = s2.device

    # build features
    f = build_pixel_features(s2, add_coords=add_coords, coord_scale=coord_scale)  # (B,F,H,W)
    B, Fch, H, W = f.shape

    # normalize barrier to [0,1]
    b = _norm01(barrier[:, :1])

    # seeds
    seeds = pick_seeds_from_s2(s2, num_slots=num_slots, min_sep=seed_min_sep)  # (B,S,2)

    # initialize protos from seeds
    protos = []
    for bb in range(B):
        pts = []
        for s in range(num_slots):
            y, x = seeds[bb, s]
            pts.append(f[bb, :, y, x])
        protos.append(torch.stack(pts, dim=0))
    protos = torch.stack(protos, dim=0)  # (B,S,F)

    # optional downsample for Dijkstra grid
    if dijkstra_down > 1:
        Hd = H // dijkstra_down
        Wd = W // dijkstra_down
        f_d = F.interpolate(f, size=(Hd, Wd), mode="bilinear", align_corners=False)
        b_d = F.interpolate(b, size=(Hd, Wd), mode="bilinear", align_corners=False)
    else:
        Hd, Wd = H, W
        f_d, b_d = f, b

    # flatten features on dijkstra grid
    # (B,F,Hd,Wd)

    dist_all = torch.zeros((B, num_slots, Hd, Wd), dtype=torch.float32, device="cpu")
    labels_all = torch.zeros((B, Hd, Wd), dtype=torch.int64, device="cpu")

    for it in range(em_iters):
        # --- E-step: build step cost grid and run multi-source Dijkstra per image ---
        for bb in range(B):
            # step cost starts from barrier
            step = 1.0 + w_barrier * b_d[bb, 0].detach().cpu().float()  # (Hd,Wd)

            # also add feature mismatch to *current best proto* locally (cheap approximation)
            # compute per-slot feature distance, then take min (winner) to bias growth
            fd = f_d[bb].detach().cpu().float()  # (F,Hd,Wd)
            fd_hw = fd.view(Fch, -1).t().contiguous()  # (Hd*Wd,F)
            p = protos[bb].detach().cpu().float()       # (S,F)
            d_feat = ((fd_hw.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(-1)  # (HW,S)
            dmin = d_feat.min(dim=1).values.view(Hd, Wd)  # (Hd,Wd)
            step = step + w_feat * _norm01(dmin).cpu().float()

            # seeds mapped to downsample grid
            sd = seeds[bb].clone()
            sd[:, 0] = torch.clamp(sd[:, 0] // dijkstra_down, 0, Hd - 1)
            sd[:, 1] = torch.clamp(sd[:, 1] // dijkstra_down, 0, Wd - 1)

            best, lab = _geodesic_multisource_dijkstra(step, sd.cpu())
            labels_all[bb] = lab
            # produce per-slot distance maps: approximate by best + per-slot feature distance (soft competition)
            # This is a pragmatic compromise: exact S Dijkstra runs would be slower.
            # It still enforces barriers strongly because "best" came from barrier-constrained propagation.
            for s in range(num_slots):
                # distance = best + feature distance to slot proto
                ds = d_feat[:, s].view(Hd, Wd)
                dist_all[bb, s] = best + 0.35 * _norm01(ds)

        # --- soft masks from distances ---
        dist_t = dist_all.to(device=device, dtype=torch.float32)  # back to torch device
        # softmax over slots
        masks_d = torch.softmax(-dist_t / max(tau, eps), dim=1)  # (B,S,Hd,Wd)

        # --- M-step: update protos using soft masks ---
        # upsample masks to full res if needed
        if dijkstra_down > 1:
            masks = F.interpolate(masks_d, size=(H, W), mode="bilinear", align_corners=False)
        else:
            masks = masks_d

        # renormalize
        masks = masks / (masks.sum(dim=1, keepdim=True) + eps)

        # update protos
        for bb in range(B):
            for s in range(num_slots):
                w = masks[bb, s:s+1]  # (1,H,W)
                denom = w.sum() + eps
                proto = (f[bb] * w).view(Fch, -1).sum(dim=1) / denom
                protos[bb, s] = proto

    # final outputs at full res
    if dijkstra_down > 1:
        dist_full = F.interpolate(dist_all.to(device=device), size=(H, W), mode="bilinear", align_corners=False)
        masks_full = F.interpolate(masks_d, size=(H, W), mode="bilinear", align_corners=False)
        labels_full = F.interpolate(labels_all.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).long()
    else:
        dist_full = dist_all.to(device=device)
        masks_full = masks_d
        labels_full = labels_all.to(device=device)

    masks_full = masks_full / (masks_full.sum(dim=1, keepdim=True) + eps)

    return GeoSlotsOut(
        masks=masks_full,
        labels=labels_full,
        protos=protos,
        dist=dist_full,
    )

