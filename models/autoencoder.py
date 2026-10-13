from __future__ import annotations

import torch
import torch.nn as nn


class OutflowAnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = max(8, input_dim * 2)
        bottleneck = max(3, input_dim // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_errors(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.mean((x - model(x)) ** 2, dim=1)
