import torch
import torch.nn.functional as F


def sym_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1)
    q = q.clamp(eps, 1)
    return (p * (p.log() - q.log())).mean() + \
           (q * (q.log() - p.log())).mean()


def consistency_loss(s2_a, s2_b_warp):
    return sym_kl(s2_a, s2_b_warp)

