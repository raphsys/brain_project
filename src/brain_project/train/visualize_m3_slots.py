# FILE: src/brain_project/train/visualize_m3_slots.py
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from brain_project.modules.m3_invariance.slots_kmeans import soft_kmeans_slots_barrier
from brain_project.modules.m3_invariance.geodesic_slots import geodesic_em_slots


# ---------------------------
# Utils I/O
# ---------------------------

def load_image(path: str, size: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size))
    x = (np.array(im).astype(np.float32) / 255.0)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return x


def gradient_barrier(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: (B,1,H,W) in [0,1]
    returns barrier in [0,1], same shape
    """
    dy = torch.abs(x01[:, :, 1:, :] - x01[:, :, :-1, :])
    dx = torch.abs(x01[:, :, :, 1:] - x01[:, :, :, :-1])

    dy = F.pad(dy, (0, 0, 1, 0))
    dx = F.pad(dx, (1, 0, 0, 0))

    g = dx + dy
    g = g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return g


# ---------------------------
# M3.5 - Lateral inhibition
# ---------------------------

def _gaussian_kernel2d(ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
    assert ksize % 2 == 1, "ksize must be odd"
    ax = torch.arange(ksize, device=device) - (ksize // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def _blur_per_channel(x: torch.Tensor, ksize: int = 11, sigma: float = 2.5) -> torch.Tensor:
    """
    x: (B,K,H,W)
    depthwise gaussian blur
    """
    B, K, H, W = x.shape
    kernel = _gaussian_kernel2d(ksize, sigma, x.device).view(1, 1, ksize, ksize)
    weight = kernel.repeat(K, 1, 1, 1)  # (K,1,ks,ks)
    return F.conv2d(x, weight, padding=ksize // 2, groups=K)


def lateral_inhibition(
    masks: torch.Tensor,
    iters: int = 12,
    alpha: float = 1.4,
    beta: float = 0.10,
    blur_ksize: int = 11,
    blur_sigma: float = 2.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    masks: (B,K,H,W) non-negative (ideally sum_K ~ 1)
    Returns sharpened masks with competition (M3.5).
    """
    x = masks.clamp_min(0.0)
    x = x / (x.sum(dim=1, keepdim=True) + eps)

    for _ in range(iters):
        x_blur = _blur_per_channel(x, ksize=blur_ksize, sigma=blur_sigma)
        comp = x_blur.sum(dim=1, keepdim=True) - x_blur
        x = x - alpha * comp + beta * x_blur
        x = F.relu(x)
        x = x / (x.sum(dim=1, keepdim=True) + eps)

    return x


def labels_from_masks(masks: torch.Tensor) -> torch.Tensor:
    return masks.argmax(dim=1)


# ---------------------------
# S2 ENRICHED (bio)
# ---------------------------

def build_s2_enriched(x: torch.Tensor, depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Perceptual S2 low-level from image + depth.
    x: (B,3,H,W) in [0,1]
    depth: (B,1,H,W) in [0,1]
    return: (B,6,H,W) normalized, sum_C=1
    """
    img_gray = x.mean(dim=1, keepdim=True)

    gx = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    gy = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])
    gx = F.pad(gx, (1, 0, 0, 0))
    gy = F.pad(gy, (0, 0, 1, 0))
    texture = gx + gy

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    s2 = torch.cat([img_gray, texture, r, g, b, depth], dim=1)

    # contrast normalize per channel
    s2 = s2 - s2.mean(dim=(2, 3), keepdim=True)
    s2 = s2 / (s2.std(dim=(2, 3), keepdim=True) + eps)

    # positive firing rates
    s2 = torch.relu(s2)

    # energy normalization (mixture-like)
    s2 = s2 / (s2.sum(dim=1, keepdim=True) + eps)
    return s2


# ---------------------------
# S2 DINOv2 (semantic)
# ---------------------------

def build_s2_from_dinov2(
    x_rgb01: torch.Tensor,          # (B,3,H,W) in [0,1]
    out_size: int = 256,
    num_channels: int = 24,
    model_name: str = "dinov2_vits14",
    seed: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Semantic-ish S2 via DINOv2 patch tokens -> projection -> softmax.
    Returns (B,K,H,W) with sum_K=1.
    """
    device = x_rgb01.device
    B, _, H, W = x_rgb01.shape

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # load model via torch.hub (internet on first run)
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.eval().to(device)

    patch = 14
    try:
        ps = model.patch_embed.patch_size
        patch = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
    except Exception:
        pass

    H2 = max((H // patch) * patch, patch)
    W2 = max((W // patch) * patch, patch)

    xin = F.interpolate(x_rgb01, size=(H2, W2), mode="bilinear", align_corners=False)
    xin = (xin - mean) / std

    with torch.no_grad():
        feats = model.forward_features(xin)
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            pt = feats["x_norm_patchtokens"]  # (B,N,C)
        elif isinstance(feats, dict) and "x_patchtokens" in feats:
            pt = feats["x_patchtokens"]
        else:
            pt = feats

    if pt.ndim != 3:
        raise RuntimeError(f"DINOv2 patch tokens expected (B,N,C), got {tuple(pt.shape)}")

    h = H2 // patch
    w = W2 // patch
    Bp, N, C = pt.shape
    if N != h * w:
        hw = int(np.sqrt(N))
        if hw * hw == N:
            h = w = hw
        elif N % h == 0:
            w = N // h
        else:
            raise RuntimeError(f"Cannot reshape patch tokens: N={N}, expected {h*w}")

    fmap = pt.transpose(1, 2).contiguous().view(B, C, h, w)
    fmap = fmap / (fmap.norm(dim=1, keepdim=True) + eps)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    Wproj = torch.randn((num_channels, C), generator=g, device=device) / np.sqrt(C)
    proj = torch.einsum("bchw,kc->bkhw", fmap, Wproj)
    proj = F.relu(proj)
    proj = F.interpolate(proj, size=(out_size, out_size), mode="bilinear", align_corners=False)

    s2 = torch.softmax(proj, dim=1)
    return s2


# ---------------------------
# MAIN
# ---------------------------

@torch.no_grad()
def main():
    out_root = Path("./runs/m3_slots")
    out_root.mkdir(parents=True, exist_ok=True)

    debug_root = Path("./runs/m3_debug")
    debug_root.mkdir(parents=True, exist_ok=True)

    path = os.environ.get("M3_IMG")
    if path is None:
        raise RuntimeError("Set M3_IMG env variable, e.g. M3_IMG=./data/real_images/xxx.jpg")

    use_dino = os.environ.get("M3_USE_DINO", "1").strip() != "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Using image:", path)

    # ------------------------------------------------
    # 0) Image
    # ------------------------------------------------
    x = load_image(path, size=256).to(device)  # (1,3,256,256)

    # ------------------------------------------------
    # 1) MiDaS depth
    # ------------------------------------------------
    print("Loading MiDaS…")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).eval().to(device)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    img_np = np.array(Image.open(path).convert("RGB")).astype(np.float32)  # H,W,3
    t_out = transform(img_np)

    # transform can return dict or tensor
    if isinstance(t_out, dict):
        inp = t_out.get("image", None)
        if inp is None:
            # fallback: take first value
            inp = next(iter(t_out.values()))
    else:
        inp = t_out

    # inp can be (C,H,W) or (B,C,H,W)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)

    inp = inp.to(device)

    depth = midas(inp)  # (B,H',W') or (B,1,H',W') depending
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    depth = F.interpolate(depth, size=(256, 256), mode="bilinear", align_corners=False)
    depth = depth / (depth.amax(dim=(2, 3), keepdim=True) + 1e-6)
    depth = depth.clamp(0.0, 1.0)

    # ------------------------------------------------
    # 2) S2 fusion (ENRICHED + DINO)
    # ------------------------------------------------
    # 2a) S2 perceptif bas niveau
    s2_low = build_s2_enriched(x, depth)  # (B,6,H,W)

    if use_dino:
        try:
            # 2b) S2 sémantique via DINOv2
            s2_dino = build_s2_from_dinov2(
                x,
                out_size=256,
                num_channels=24,
                model_name="dinov2_vits14",
                seed=0,
            )
            # 2c) fusion
            s2 = torch.cat([s2_low, s2_dino], dim=1)  # (B,30,H,W)
            s2 = s2 / (s2.sum(dim=1, keepdim=True) + 1e-6)
        except Exception as e:
            print("[WARN] DINOv2 failed, fallback to s2_low only. Error:", repr(e))
            s2 = s2_low
    else:
        s2 = s2_low

    print("S2:", tuple(s2.shape), "min/max:", float(s2.min()), float(s2.max()))

    # ------------------------------------------------
    # 3) Barrier = |∇depth|
    # ------------------------------------------------
    barrier = gradient_barrier(depth)

    # ------------------------------------------------
    # 4) M3.2 soft-kmeans (baseline)
    # ------------------------------------------------
    out_km = soft_kmeans_slots_barrier(
        s2,
        barrier=barrier,
        num_slots=6,
        iters=15,
        tau=0.25,
        seed=0,
    )
    km_masks = out_km.masks

    # ------------------------------------------------
    # 5) M3.4 + M3.6 geodesic EM (DEBUG LOURD)
    # ------------------------------------------------
    # base RGB for seeds overlay (CPU numpy)
    base_rgb = x[0].detach().cpu().permute(1, 2, 0).numpy()

    run_id = Path(path).stem
    dbg_dir = debug_root / run_id
    dbg_dir.mkdir(parents=True, exist_ok=True)

    out_geo = geodesic_em_slots(
        s2,
        barrier=barrier,
        num_slots=6,
        em_iters=6,
        tau=0.20,
        w_barrier=6.0,
        w_feat=1.0,
        dijkstra_down=1,
        seed_min_sep=14,
        seed_mode="perceptual",
        seed=0,

        debug_dir=dbg_dir,
        debug_prefix="m34",
        debug_every=1,
        debug_save_npy=True,
        debug_save_png=True,
        debug_base_rgb01=base_rgb,
    )

    seeds = out_geo.seeds
    geo_masks = out_geo.masks
    geo_labels = out_geo.labels

    # ------------------------------------------------
    # 6) M3.5 lateral inhibition
    # ------------------------------------------------
    m35_masks = lateral_inhibition(
        geo_masks,
        iters=14,
        alpha=1.6,
        beta=0.12,
        blur_ksize=11,
        blur_sigma=2.2,
    )
    m35_labels = labels_from_masks(m35_masks)

    # ------------------------------------------------
    # 7) Visualization
    # ------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    axes[0, 0].imshow(base_rgb)
    axes[0, 0].set_title("Image")

    axes[0, 1].imshow(depth[0, 0].detach().cpu(), cmap="gray")
    axes[0, 1].set_title("Depth")

    axes[0, 2].imshow(barrier[0, 0].detach().cpu(), cmap="gray")
    axes[0, 2].set_title("|∇depth|")

    axes[0, 3].imshow(km_masks[0, 0].detach().cpu(), cmap="gray")
    axes[0, 3].set_title("M3.2 slot[0]")

    axes[1, 0].imshow(geo_masks[0, 0].detach().cpu(), cmap="gray")
    axes[1, 0].set_title("M3.4 slot[0]")

    axes[1, 1].imshow(geo_labels[0].detach().cpu())
    axes[1, 1].set_title("M3.4 labels")

    axes[1, 2].imshow(m35_masks[0, 0].detach().cpu(), cmap="gray")
    axes[1, 2].set_title("M3.5 slot[0]")

    axes[1, 3].imshow(m35_labels[0].detach().cpu())
    axes[1, 3].set_title("M3.5 labels")

    for ax in axes.flatten():
        ax.axis("off")

    # Seeds overlay (on image)
    ax = axes[0, 0]
    colors = plt.cm.tab10.colors
    for s, (yy, xx) in enumerate(seeds[0].detach().cpu().numpy()):
        ax.scatter(xx, yy, c=[colors[s % len(colors)]], s=80, marker="x", linewidths=2)
        ax.text(xx + 3, yy + 3, f"{s}", color=colors[s % len(colors)], fontsize=9, weight="bold")
    ax.set_title("Image + seeds")

    plt.tight_layout()
    out_path = out_root / "m3_slots_m35.png"
    plt.savefig(out_path, dpi=170)
    print("Saved:", out_path)
    print("Debug saved in:", dbg_dir)


if __name__ == "__main__":
    main()

