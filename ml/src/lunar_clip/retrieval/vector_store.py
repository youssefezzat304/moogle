from __future__ import annotations

import torch


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.vectors: torch.Tensor | None = None
        self.ids: list[int | str] = []

    def add(self, ids: list[int | str], vectors: torch.Tensor) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D. Got {tuple(vectors.shape)}.")
        self.ids.extend(ids)
        vectors = vectors.detach().cpu()
        self.vectors = vectors if self.vectors is None else torch.cat([self.vectors, vectors])

    def search(self, query: torch.Tensor, top_k: int = 10) -> list[tuple[int | str, float]]:
        if self.vectors is None:
            return []
        if query.ndim == 1:
            query = query.unsqueeze(0)
        scores = query.detach().cpu() @ self.vectors.t()
        values, indices = scores[0].topk(min(top_k, scores.shape[1]))
        return [(self.ids[int(index)], float(value)) for value, index in zip(values, indices)]
