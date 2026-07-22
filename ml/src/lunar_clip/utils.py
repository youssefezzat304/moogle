from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.retrieval.vector_store import InMemoryVectorStore

# TODO: Consolidate vision backend registration in one registry shared by
# configuration parsing and LunarVisionEncoder. The supported encoder list,
# modality mapping, checkpoint loader, and input contract should not diverge.
VISION_ENCODERS = ("geo", "wac","fusion")


def build_in_memory_index(
    ids: list[int | str],
    vectors: torch.Tensor,
) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(ids=ids, vectors=vectors)
    return store


def search_index(
    store: InMemoryVectorStore,
    query: torch.Tensor,
    top_k: int = 10,
) -> list[tuple[int | str, float]]:
    return store.search(query=query, top_k=top_k)


def build_vision_adapter(config: dict[str, Any]):
    encoder = config.get("encoder") or config.get("adapter")
    if not encoder:
        raise ValueError("Vision config must contain 'encoder' or 'adapter'.")
    return LunarVisionEncoder(
        encoder=str(encoder),
        checkpoint_path=config.get("checkpoint_path"),
        freeze_encoder=bool(config.get("freeze_encoder", False)),
        device=config.get("device"),
    )


def build_projection_head(input_dim: int, projection_dim: int) -> nn.Linear:
    if input_dim <= 0:
        raise ValueError("input_dim must be greater than 0.")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be greater than 0.")
    return nn.Linear(input_dim, projection_dim, bias=False)
