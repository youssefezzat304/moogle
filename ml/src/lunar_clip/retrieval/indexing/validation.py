from __future__ import annotations

from pathlib import Path

import torch

from lunar_clip.retrieval.indexing.artifacts import (
    IndexArtifact,
    IndexManifest,
    load_index_artifact,
)
from lunar_clip.retrieval.indexing.contracts import EmbeddingDescriptor
from lunar_data.catalog.metadata import CatalogArtifact


def validate_index_tensors(
    *,
    patch_ids: torch.Tensor,
    embeddings: torch.Tensor,
    manifest: IndexManifest,
    catalog_patch_ids: list[int],
) -> None:
    if patch_ids.ndim != 1 or patch_ids.dtype != torch.int64:
        raise ValueError("Index patch_ids must be a one-dimensional int64 tensor.")
    if embeddings.ndim != 2 or embeddings.dtype != torch.float32:
        raise ValueError("Index embeddings must be a two-dimensional float32 tensor.")
    if embeddings.shape != (
        manifest.index_size,
        manifest.descriptor.embedding_dimension,
    ):
        raise ValueError("Embedding shape does not match the index manifest.")
    if patch_ids.shape[0] != manifest.index_size:
        raise ValueError("Patch ID count does not match the index manifest.")
    if patch_ids.tolist() != catalog_patch_ids:
        raise ValueError("Index patch IDs must match the catalog row-for-row.")
    if len(set(patch_ids.tolist())) != patch_ids.numel():
        raise ValueError("Index patch IDs must be unique.")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Index embeddings must all be finite.")
    if manifest.descriptor.normalized:
        norms = torch.linalg.vector_norm(embeddings, dim=1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("Index embeddings must have unit norm.")


def validate_index_artifact(
    root: str | Path,
    *,
    catalog: CatalogArtifact,
) -> IndexArtifact:
    artifact = load_index_artifact(root)
    if artifact.manifest.catalog_id != catalog.manifest.catalog_id:
        raise ValueError("Index references a different catalog.")
    if artifact.manifest.index_size != catalog.manifest.index_size:
        raise ValueError("Index and catalog sizes differ.")
    validate_index_tensors(
        patch_ids=artifact.patch_ids,
        embeddings=artifact.embeddings,
        manifest=artifact.manifest,
        catalog_patch_ids=catalog.patch_ids,
    )
    return artifact


def validate_index_compatibility(
    artifact: IndexArtifact,
    descriptor: EmbeddingDescriptor,
) -> None:
    if artifact.manifest.descriptor != descriptor:
        raise ValueError(
            "Index embedding pipeline does not match the configured runtime model."
        )
