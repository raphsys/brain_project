import torch


def warp_inverse(tensor, grid):
    """
    tensor: (B,K,H,W)
    grid: affine grid used to generate the view

    Applies inverse warp using same grid.
    """
    return torch.nn.functional.grid_sample(
        tensor, grid, align_corners=False
    )

