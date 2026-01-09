from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from brain_project.train.utils import load_image_tensor

# ===== M1 =====
from brain_project.modules.m1_perception.soft_regions import M1V1V2Perception

# ===== M2 (classes explicites) =====
from brain_project.modules.m2_grouping.end_stopping import EndStopping
from brain_project.modules.m2_grouping.cocircularity import CoCircularity
from brain_project.modules.m2_grouping.region_stabilization import RegionStabilization

# ===== M3 =====
from brain_project.modules.m3_invariance.geodesic_slots import geodesic_em_slots


OUT = Path("./runs/m3_slots")
OUT.mkdir(parents=True, exist_ok=True)


def gradient_edge(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,K,H,W) -> edge: (B,1,H,W)
    """
    dx = x[..., 1:, :] - x[..., :-1, :]       # (B,K,H-1,W)
    dx = F.pad(dx, (0, 0, 0, 1))              # -> (B,K,H,W)

    dy = x[..., :, 1:] - x[..., :, :-1]       # (B,K,H,W-1)
    dy = F.pad(dy, (0, 1, 0, 0))              # -> (B,K,H,W)

    g = torch.sqrt(dx.pow(2) + dy.pow(2))     # (B,K,H,W)
    g = g.mean(1, keepdim=True)               # (B,1,H,W)
    g = g / (g.max() + 1e-6)
    return g


def unwrap_tensor(out):
    """
    Convertit les objets *Out* (EndStoppingOut, etc.) en tenseur.
    Stratégie: champs courants, sinon dict/tuple, sinon erreur explicite.
    """
    if isinstance(out, torch.Tensor):
        return out

    # champs fréquents
    for k in ("edge", "edges", "out", "map", "resp", "response", "tensor", "x"):
        if hasattr(out, k):
            v = getattr(out, k)
            if isinstance(v, torch.Tensor):
                return v

    # dict
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v

    # tuple/list
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return v

    raise TypeError(
        f"Impossible d'extraire un Tensor depuis {type(out)}.\n"
        f"Attributs disponibles: {dir(out)[:40]} ..."
    )


@torch.no_grad()
def main():
    img_path = "./data/real_images/test/5805a4ae-64f4-4d98-be10-612e0483f1fe.jpg"
    print("Using image:", img_path)

    x = load_image_tensor(img_path)

    # ========= M1 =========
    m1 = M1V1V2Perception()
    out1 = m1(x)
    s1 = out1.s1
    depth = out1.depth

    # ========= Edge =========
    edge = gradient_edge(s1)                  # (B,1,H,W)

    # ========= M2 (pipeline explicite) =========
    m2_es = EndStopping(n_bins=8)
    m2_cc = CoCircularity()                   # si ton __init__ demande des args, on ajustera
    m2_rs = RegionStabilization()             # idem

    e_out = m2_es(edge)
    e = unwrap_tensor(e_out)                  # <<<<< FIX: extraire un Tensor pour cc

    c_out = m2_cc(e)
    c = unwrap_tensor(c_out)                  # <<<<< idem pour rs

    s2_out = m2_rs(c)
    s2 = unwrap_tensor(s2_out)

    # ========= M3 (géodésique) =========
    out = geodesic_em_slots(
        s2,
        barrier=s2,
        num_slots=6,
        em_iters=5,
        tau=0.20,
        w_barrier=6.0,
        w_feat=1.0,
        dijkstra_down=1,
        seed_min_sep=12,
    )

    masks = out.masks[0]
    labels = out.labels[0]

    # ========= VISU =========
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    axes[0, 0].imshow(x[0].permute(1, 2, 0))
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(depth[0, 0], cmap="gray")
    axes[0, 1].set_title("MiDaS depth")

    axes[0, 2].imshow(s1.argmax(1)[0])
    axes[0, 2].set_title("M1 regions (S1)")

    axes[0, 3].imshow(edge[0, 0], cmap="gray")
    axes[0, 3].set_title("Edge map")

    axes[1, 0].imshow(s2.argmax(1)[0])
    axes[1, 0].set_title("M2 regions (S2)")

    axes[1, 1].imshow(labels)
    axes[1, 1].set_title("M3 labels")

    for i in range(4):
        axes[2, i].imshow(masks[i], cmap="gray")
        axes[2, i].set_title(f"Slot {i}")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT / "m3_slots.png"
    plt.savefig(out_path, dpi=160)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

