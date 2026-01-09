import torch
import glob, os, numpy as np
from PIL import Image

from brain_project.modules.m3_invariance.m3_refiner import M3Refiner
from brain_project.modules.m3_invariance.m3_consistency import consistency_loss
from brain_project.modules.m3_invariance.augment import *
from brain_project.modules.m3_invariance.warp import warp_inverse

CACHE_DIR = "./runs/m3_cache"


def main():

    test_cache = f"{CACHE_DIR}/test.pt"
    ckpt = "./runs/m3/refiner_last.pth"

    print("\n=== M3 EVAL (FAST) ===")

    if not os.path.exists(test_cache):
        raise RuntimeError("Run train first to build cache")

    S2 = torch.load(test_cache)

    refiner = M3Refiner(k=8)
    refiner.load_state_dict(torch.load(ckpt, map_location="cpu"))
    refiner.eval()

    losses = []

    for i in range(len(S2)):

        s2 = S2[i:i+1]

        rot, tx, ty, sc = random_affine_params(
            max_rot=10, max_trans=0.08, max_scale=0.08
        )
        A = affine_matrix(rot, tx, ty, sc)

        grid = torch.nn.functional.affine_grid(
            A.unsqueeze(0), s2.size(), align_corners=False
        )

        s2b = torch.nn.functional.grid_sample(
            s2, grid, align_corners=False
        )

        s2a_r = refiner(s2)
        s2b_r = refiner(s2b)

        s2b_w = warp_inverse(s2b_r, grid)

        loss = consistency_loss(s2a_r, s2b_w)
        losses.append(loss.item())

    print("Mean test loss:", np.mean(losses))


if __name__ == "__main__":
    main()

