from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RetrievalBatch:
    vectors: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(
                "RetrievalBatch.vectors must have shape (batch_size, hidden_dim). "
                f"Got {tuple(self.vectors.shape)}."
            )
