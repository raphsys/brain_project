import torch
import math
import random
import torchvision.transforms.functional as TF


def random_affine_params(max_rot=8, max_trans=0.05, max_scale=0.05):
    rot = random.uniform(-max_rot, max_rot)
    tx = random.uniform(-max_trans, max_trans)
    ty = random.uniform(-max_trans, max_trans)
    sc = 1.0 + random.uniform(-max_scale, max_scale)
    return rot, tx, ty, sc


def affine_matrix(rot_deg, tx, ty, scale):
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)

    A = torch.tensor([
        [ scale*c, -scale*s, tx ],
        [ scale*s,  scale*c, ty ]
    ], dtype=torch.float32)

    return A


def apply_affine(x, A):
    """
    x: (B,3,H,W)
    A: (2,3)
    """
    B, C, H, W = x.shape
    A = A.unsqueeze(0).to(x.device)

    grid = torch.nn.functional.affine_grid(A, size=x.size(), align_corners=False)
    y = torch.nn.functional.grid_sample(x, grid, align_corners=False)
    return y, grid

