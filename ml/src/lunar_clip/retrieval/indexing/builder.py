from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from lunar_clip.retrieval.indexing.artifacts import (
    IndexManifest,
    default_index_id,
    write_index_files,
)
from lunar_clip.retrieval.indexing.contracts import ImageBatch, ImageEmbedder
from lunar_clip.retrieval.indexing.validation import (
    validate_index_artifact,
    validate_index_tensors,
)
from lunar_data.catalog.metadata import CatalogArtifact
from lunar_data.catalog.validation import validate_catalog_frame


@dataclass(frozen=True)
class IndexBuildConfig:
    index_id: str | None = None

    def __post_init__(self) -> None:
        if self.index_id is not None and not self.index_id.strip():
            raise ValueError("index_id must be non-empty when provided.")


def build_index(
    *,
    catalog: CatalogArtifact,
    batches: Iterable[ImageBatch],
    embedder: ImageEmbedder,
    output: str | Path,
    config: IndexBuildConfig = IndexBuildConfig(),
) -> Path:
    """Encode deterministic image batches and atomically publish an index."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"Index output already exists: {output_path}")
    validate_catalog_frame(
        catalog.metadata,
        manifest=catalog.manifest,
        root=catalog.root,
    )

    patch_id_parts: list[torch.Tensor] = []
    embedding_parts: list[torch.Tensor] = []
    for batch in batches:
        embeddings = embedder.encode(batch.images)
        if embeddings.ndim != 2:
            raise ValueError("Image embedder must return a two-dimensional tensor.")
        if embeddings.shape[0] != batch.patch_ids.shape[0]:
            raise ValueError("Image embedder returned the wrong batch size.")
        patch_id_parts.append(batch.patch_ids.detach().cpu())
        embedding_parts.append(embeddings.detach().cpu().to(torch.float32))

    if not patch_id_parts:
        raise ValueError("Cannot build an index from zero batches.")
    patch_ids = torch.cat(patch_id_parts)
    embeddings = torch.cat(embedding_parts)
    descriptor = embedder.descriptor
    manifest = IndexManifest(
        index_id=config.index_id or default_index_id(descriptor),
        catalog_id=catalog.manifest.catalog_id,
        index_size=catalog.manifest.index_size,
        descriptor=descriptor,
    )
    validate_index_tensors(
        patch_ids=patch_ids,
        embeddings=embeddings,
        manifest=manifest,
        catalog_patch_ids=catalog.patch_ids,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
        )
    )
    try:
        write_index_files(
            temporary,
            manifest=manifest,
            patch_ids=patch_ids,
            embeddings=embeddings,
        )
        validate_index_artifact(temporary, catalog=catalog)
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path
