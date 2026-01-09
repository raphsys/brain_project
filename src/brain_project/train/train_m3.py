import torch
import torch.optim as optim
import os, time, glob
import numpy as np
from PIL import Image

from brain_project.modules.m3_invariance.m3_refiner import M3Refiner
from brain_project.modules.m3_invariance.m3_consistency import consistency_loss
from brain_project.modules.m3_invariance.augment import *
from brain_project.modules.m3_invariance.warp import warp_inverse

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception
from brain_project.modules.m2_grouping import CoCircularity, stabilize_regions


CACHE_DIR = "./runs/m3_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------- LOAD IMAGES ----------------
def load_images(folder, size=256):
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size))
        x = torch.from_numpy(np.array(im)).float()/255.
        imgs.append(x.permute(2,0,1))
    return torch.stack(imgs)


# ---------------- BUILD CACHE ----------------
@torch.no_grad()
def build_cache(folder, name):

    cache_file = f"{CACHE_DIR}/{name}.pt"
    if os.path.exists(cache_file):
        print("✔ Cache exists:", cache_file)
        return torch.load(cache_file)

    print(">> Building cache:", name)

    X = load_images(folder)

    m1 = M1V1V2Perception(k_regions=8, use_depth=True).eval()
    m2 = CoCircularity()

    S2_all = []

    for i in range(len(X)):
        x = X[i:i+1]

        out1 = m1(x)
        c = m2(out1.boundary)

        s2 = stabilize_regions(
            out1.s1, out1.boundary,
            c.completed, out1.depth
        ).s2

        S2_all.append(s2.cpu())

    S2_all = torch.cat(S2_all)
    torch.save(S2_all, cache_file)

    print("✔ Cache saved:", cache_file)
    return S2_all


# ---------------- MAIN ----------------
def main():

    train_dir = "./data/real_images/train"
    test_dir  = "./data/real_images/test"
    epochs = 10
    lr = 1e-3

    print("\n=== M3 TRAIN (FAST) ===")

    S2 = build_cache(train_dir, "train")
    S2_test  = build_cache(test_dir, "test")

    refiner = M3Refiner(k=8)
    opt = optim.Adam(refiner.parameters(), lr=lr)

    for ep in range(epochs):

        losses = []
        t0 = time.time()

        for i in range(len(S2)):

            s2 = S2[i:i+1]

            # augmentation spatiale SUR S2
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

            # refine
            s2a_r = refiner(s2)
            s2b_r = refiner(s2b)

            s2b_w = warp_inverse(s2b_r, grid)

            loss = consistency_loss(s2a_r, s2b_w)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[Epoch {ep+1:02d}] "
              f"loss={np.mean(losses):.4f} "
              f"time={time.time()-t0:.1f}s")

    os.makedirs("./runs/m3", exist_ok=True)
    torch.save(refiner.state_dict(), "./runs/m3/refiner_last.pth")
    print("✔ Saved: ./runs/m3/refiner_last.pth")


if __name__ == "__main__":
    main()

