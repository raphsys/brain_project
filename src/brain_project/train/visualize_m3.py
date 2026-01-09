import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception
from brain_project.modules.m2_grouping import CoCircularity, stabilize_regions

from brain_project.modules.m3_invariance.augment import (
    random_affine_params, affine_matrix, apply_affine
)
from brain_project.modules.m3_invariance.warp import warp_inverse
from brain_project.modules.m3_invariance.m3_consistency import consistency_loss


def load_image(path, size=256):
    im = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(im)).float()/255.
    return x.permute(2,0,1).unsqueeze(0)


def main():

    # ====== Charger même dossier que précédemment ======
    img_dir = "./data/real_images"
    paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))

    if len(paths) == 0:
        raise RuntimeError("Aucune image trouvée dans ./data/real_images")

    path = paths[0]   # même image que M1/M2
    print("Using image:", path)

    x = load_image(path)

    # ====== Augmentation ======
    rot, tx, ty, sc = random_affine_params()
    A = affine_matrix(rot, tx, ty, sc)
    xa, grid = apply_affine(x, A)

    # ====== Pipeline M1 + M2 ======
    m1 = M1V1V2Perception(k_regions=8, use_depth=True).eval()

    out1a = m1(x)
    out1b = m1(xa)

    m2 = CoCircularity()
    c_a = m2(out1a.boundary)
    c_b = m2(out1b.boundary)

    s2a = stabilize_regions(
        out1a.s1, out1a.boundary, c_a.completed, out1a.depth
    ).s2

    s2b = stabilize_regions(
        out1b.s1, out1b.boundary, c_b.completed, out1b.depth
    ).s2

    # ====== Warp inverse ======
    s2b_w = warp_inverse(s2b, grid)

    loss = consistency_loss(s2a, s2b_w)
    print("Consistency loss:", loss.item())

    # ====== Visualisation ======
    plt.figure(figsize=(16,8))

    plt.subplot(2,4,1)
    plt.imshow(x[0].permute(1,2,0))
    plt.title("Original")

    plt.subplot(2,4,2)
    plt.imshow(xa[0].permute(1,2,0))
    plt.title("Augmented")

    plt.subplot(2,4,3)
    plt.imshow(torch.argmax(s2a[0],0), cmap="tab10")
    plt.title("S2 original")

    plt.subplot(2,4,4)
    plt.imshow(torch.argmax(s2b[0],0), cmap="tab10")
    plt.title("S2 augmented")

    plt.subplot(2,4,5)
    plt.imshow(torch.argmax(s2b_w[0],0), cmap="tab10")
    plt.title("Warped back")

    plt.subplot(2,4,6)
    plt.imshow((s2a[0,0]-s2b_w[0,0]).abs(), cmap="hot")
    plt.title("|Δ region0|")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

