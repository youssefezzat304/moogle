from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

from lunar_clip.model.loading import (
    PromotedModelManifest,
    load_promoted_lunar_clip_model,
)
from lunar_clip.retrieval.indexing import (
    EmbeddingDescriptor,
    validate_index_artifact,
)
from lunar_data.catalog import validate_catalog_artifact
from moogle_inference.engine import RetrievalEngine


class LunarClipTextModel(Protocol):
    def eval(self) -> Any: ...

    def to(self, device: torch.device) -> Any: ...

    def encode_text(self, text_batch: list[str]) -> torch.Tensor: ...


class LunarClipTextEmbedder:
    def __init__(
        self,
        *,
        model: LunarClipTextModel,
        descriptor: EmbeddingDescriptor,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self._descriptor = descriptor
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._descriptor

    def encode(self, query: str) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            embedding = self.model.encode_text([query])
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("LunarCLIP encode_text must return a tensor.")
        return embedding.detach().cpu()


def load_promoted_text_embedder(
    manifest_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LunarClipTextEmbedder:
    loaded = load_promoted_lunar_clip_model(manifest_path)
    return LunarClipTextEmbedder(
        model=loaded.model,
        descriptor=_embedding_descriptor(loaded.manifest),
        device=device,
    )


def load_retrieval_engine(
    *,
    catalog_path: str | Path,
    index_path: str | Path,
    model_manifest_path: str | Path,
    device: str | torch.device = "cpu",
) -> RetrievalEngine:
    """Load and compatibility-check all production retrieval artifacts."""

    catalog = validate_catalog_artifact(catalog_path)
    index = validate_index_artifact(index_path, catalog=catalog)
    text_embedder = load_promoted_text_embedder(
        model_manifest_path,
        device=device,
    )
    return RetrievalEngine(
        catalog=catalog,
        index=index,
        text_embedder=text_embedder,
    )


def _embedding_descriptor(
    manifest: PromotedModelManifest,
) -> EmbeddingDescriptor:
    return EmbeddingDescriptor(
        model_id=manifest.model_id,
        checkpoint_sha256=manifest.checkpoint_sha256,
        modality=manifest.modality,
        preprocessing_id=manifest.preprocessing_id,
        embedding_dimension=manifest.projection_dim,
    )
