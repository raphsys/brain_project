import torch
import torch.nn as nn


class M3Refiner(nn.Module):
    """
    Petit réseau qui apprend à stabiliser S2
    """
    def __init__(self, k=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(k, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, k, 1)
        )

    def forward(self, s2):
        return torch.softmax(self.net(s2), dim=1)

