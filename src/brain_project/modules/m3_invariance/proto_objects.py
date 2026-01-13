# FILE: src/brain_project/modules/m3_invariance/proto_objects.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ProtoObject:
    id: int
    centroid: torch.Tensor      # (2,) in [-1,1]
    area: float
    per_embed: torch.Tensor     # (Dper,)
    sem_embed: Optional[torch.Tensor]  # (Dsem,) or None
    energy: float
    age: int


def extract_proto_objects(
    masks: torch.Tensor,      # (1,S,H,W)
    s2: torch.Tensor,         # (1,K,H,W)
    s2_dino: Optional[torch.Tensor] = None,  # (1,Kd,H,W)
    min_area: int = 50,
    eps: float = 1e-6,
) -> List[ProtoObject]:
    """
    Extract proto objects from soft masks.
    """
    assert masks.shape[0] == 1, "batch=1 for now"
    _, S, H, W = masks.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=masks.device),
        torch.linspace(-1.0, 1.0, W, device=masks.device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1,2,H,W)

    out: List[ProtoObject] = []
    pid = 0

    for s in range(S):
        w = masks[0, s:s+1]  # (1,H,W)
        area = float((w > 0.5).sum().item())
        if area < min_area:
            continue

        # centroid
        wsum = w.sum() + eps
        c = (coords[0] * w).view(2, -1).sum(dim=1) / wsum  # (2,)

        # perceptual embedding: weighted mean of s2 channels
        per = (s2[0] * masks[0, s:s+1]).view(s2.shape[1], -1).sum(dim=1) / wsum

        # semantic embedding: weighted mean of s2_dino channels
        sem = None
        if s2_dino is not None:
            sem = (s2_dino[0] * masks[0, s:s+1]).view(s2_dino.shape[1], -1).sum(dim=1) / wsum

        # energy proxy: compactness / uncertainty
        # (plus simple = area-normalized entropy)
        p = masks[0, :, :, :].clamp_min(eps)
        ent = -(p * torch.log(p)).sum(dim=0).mean()
        energy = float(ent.item())

        out.append(
            ProtoObject(
                id=pid,
                centroid=c.detach().cpu(),
                area=float(area),
                per_embed=per.detach().cpu(),
                sem_embed=None if sem is None else sem.detach().cpu(),
                energy=energy,
                age=0,
            )
        )
        pid += 1

    return out


def match_and_update(
    prev: Dict[int, ProtoObject],
    cur_list: List[ProtoObject],
    w_iou: float = 0.0,         # ici on fait sans IoU pour garder simple
    w_cent: float = 0.30,
    w_per: float = 0.40,
    w_sem: float = 0.30,
    max_cost: float = 0.65,
) -> Dict[int, ProtoObject]:
    """
    Greedy matching (stable + simple).
    Cost = w_cent*L2 + w_per*(1-cos) + w_sem*(1-cos)
    """
    def cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
        a = a / (a.norm() + eps)
        b = b / (b.norm() + eps)
        return float(1.0 - torch.dot(a, b).item())

    prev_items = list(prev.items())  # [(id,obj),...]
    used_prev = set()
    next_prev: Dict[int, ProtoObject] = {}

    next_id = (max(prev.keys()) + 1) if len(prev) > 0 else 0

    for cur in cur_list:
        best_id = None
        best_cost = 1e9

        for pid, pobj in prev_items:
            if pid in used_prev:
                continue

            dc = float(torch.norm(cur.centroid - pobj.centroid).item())
            dp = cos_dist(cur.per_embed, pobj.per_embed)

            ds = 0.0
            if cur.sem_embed is not None and pobj.sem_embed is not None:
                ds = cos_dist(cur.sem_embed, pobj.sem_embed)

            cost = w_cent * dc + w_per * dp + w_sem * ds
            if cost < best_cost:
                best_cost = cost
                best_id = pid

        if best_id is not None and best_cost <= max_cost:
            used_prev.add(best_id)
            old = prev[best_id]
            # update
            next_prev[best_id] = ProtoObject(
                id=best_id,
                centroid=cur.centroid,
                area=cur.area,
                per_embed=cur.per_embed,
                sem_embed=cur.sem_embed,
                energy=0.7 * old.energy + 0.3 * cur.energy,
                age=old.age + 1,
            )
        else:
            # new object
            nid = next_id
            next_id += 1
            next_prev[nid] = ProtoObject(
                id=nid,
                centroid=cur.centroid,
                area=cur.area,
                per_embed=cur.per_embed,
                sem_embed=cur.sem_embed,
                energy=cur.energy,
                age=0,
            )

    return next_prev

