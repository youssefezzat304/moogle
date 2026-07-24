from __future__ import annotations

from pathlib import Path

import torch
from lunar_clip.model.loading import load_promoted_lunar_clip_model
from lunar_clip.retrieval.indexing.contracts import EmbeddingDescriptor
from lunar_clip.retrieval.indexing.embedder import LunarClipImageEmbedder


def load_promoted_embedder(
    manifest_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LunarClipImageEmbedder:
    """Load any registered promoted LunarCLIP architecture for indexing."""

    loaded = load_promoted_lunar_clip_model(manifest_path)
    manifest = loaded.manifest

    descriptor = EmbeddingDescriptor(
        model_id=manifest.model_id,
        checkpoint_sha256=manifest.checkpoint_sha256,
        modality=manifest.modality,
        preprocessing_id=manifest.preprocessing_id,
        embedding_dimension=manifest.projection_dim,
    )
    return LunarClipImageEmbedder(
        model=loaded.model,
        descriptor=descriptor,
        device=device,
    )
