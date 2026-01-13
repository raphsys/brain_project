# FILE: src/brain_project/train/bench_m3.py
import os
from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image

from brain_project.modules.m3_invariance.m3_pipeline import run_m3


def load_rgb(path: str, size: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size))
    x = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def main():
    device = torch.device("cpu")
    img_path = os.environ["M3_IMG"]

    x = load_rgb(img_path, 256).to(device)

    # depth optionnel: ici on met un depth fake si pas fourni
    depth = torch.zeros((1, 1, 256, 256), device=device)

    out = run_m3(
        x,
        depth,
        num_slots=6,
        use_dino=True,
        dino_channels=24,
        seed_mode="perceptual",
    )

    # dump minimal
    out_dir = Path("runs/m3_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "img": img_path,
        "s2_channels": int(out.s2.shape[1]),
        "min_s2": float(out.s2.min().item()),
        "max_s2": float(out.s2.max().item()),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Saved:", out_dir / "stats.json")


if __name__ == "__main__":
    main()

