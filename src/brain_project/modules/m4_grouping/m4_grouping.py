# FILE: src/brain_project/modules/m4_grouping/m4_grouping.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class M4Group:
    group_id: int
    members: List[int]


def _cosine_dist(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    A = A / (A.norm(dim=1, keepdim=True) + eps)
    B = B / (B.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (A @ B.t())


def _pairwise_l2(X: torch.Tensor) -> torch.Tensor:
    # (N,D) -> (N,N)
    return torch.cdist(X, X)


def m4_grouping(
    sem: torch.Tensor,      # (N,Dsem)
    per: torch.Tensor,      # (N,Dper)
    cent: torch.Tensor,     # (N,2) normalized [-1,1]
    lam_sem: float = 0.45,
    lam_per: float = 0.45,
    lam_cent: float = 0.10,
    tau: float = 0.45,      # threshold distance
) -> List[M4Group]:
    """
    Graph-based grouping (simple agglomeration).
    - Build adjacency if D(i,j) < tau
    - Return connected components
    """
    N = sem.shape[0]
    Dsem = _cosine_dist(sem, sem)
    Dper = _cosine_dist(per, per)
    Dcen = _pairwise_l2(cent)

    D = lam_sem * Dsem + lam_per * Dper + lam_cent * Dcen
    A = (D < tau).to(torch.int64)

    # connected components
    visited = [False] * N
    groups: List[M4Group] = []
    gid = 0

    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        members = []

        while stack:
            u = stack.pop()
            members.append(u)
            neigh = torch.where(A[u] > 0)[0].tolist()
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        groups.append(M4Group(group_id=gid, members=members))
        gid += 1

    return groups

