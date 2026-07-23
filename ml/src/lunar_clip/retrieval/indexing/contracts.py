from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import torch


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EmbeddingDescriptor:
    """Identity of the complete image-embedding pipeline."""

    model_id: str
    checkpoint_sha256: str
    modality: str
    preprocessing_id: str
    embedding_dimension: int
    normalized: bool = True
    embedding_dtype: str = "float32"
    similarity_metric: str = "cosine"

    def __post_init__(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("modality", self.modality),
            ("preprocessing_id", self.preprocessing_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty.")
        if not SHA256_PATTERN.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest.")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")
        if self.embedding_dtype != "float32":
            raise ValueError("Only float32 index artifacts are supported.")
        if self.similarity_metric != "cosine":
            raise ValueError("Only cosine similarity is supported.")
        if not isinstance(self.normalized, bool):
            raise ValueError("normalized must be a boolean.")
        if not self.normalized:
            raise ValueError("Cosine indexes must contain normalized embeddings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "modality": self.modality,
            "preprocessing_id": self.preprocessing_id,
            "embedding_dimension": self.embedding_dimension,
            "embedding_dtype": self.embedding_dtype,
            "normalized": self.normalized,
            "similarity_metric": self.similarity_metric,
        }


@dataclass(frozen=True)
class ImageBatch:
    patch_ids: torch.Tensor
    images: Any

    def __post_init__(self) -> None:
        if self.patch_ids.ndim != 1:
            raise ValueError("ImageBatch.patch_ids must be one-dimensional.")
        if self.patch_ids.dtype != torch.int64:
            raise ValueError("ImageBatch.patch_ids must use torch.int64.")


class ImageEmbedder(Protocol):
    @property
    def descriptor(self) -> EmbeddingDescriptor: ...

    def encode(self, images: Any) -> torch.Tensor: ...
