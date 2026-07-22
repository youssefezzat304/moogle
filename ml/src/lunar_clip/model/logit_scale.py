from __future__ import annotations

import math

import torch
from torch import nn


class LogitScale(nn.Module):
    def __init__(self, temperature: float = 0.07, max_scale: float = 100.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        self.value = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.max_scale = max_scale

    def forward(self) -> torch.Tensor:
        return self.value.exp().clamp(max=self.max_scale)
