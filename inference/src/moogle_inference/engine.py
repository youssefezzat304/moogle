from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from lunar_clip.retrieval.indexing import (
    EmbeddingDescriptor,
    IndexArtifact,
    validate_index_compatibility,
    validate_index_tensors,
)
from lunar_data.catalog import CatalogArtifact, validate_catalog_frame


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    patch_id: int
    similarity: float
    description: str
    source_version: str
    prompt_style: str
    wac_image_path: Path
    latitude: float
    longitude: float


class TextEmbedder(Protocol):
    @property
    def descriptor(self) -> EmbeddingDescriptor: ...

    def encode(self, query: str) -> torch.Tensor: ...


class RetrievalEngine:
    """Search a validated image index and join results with catalog metadata."""

    def __init__(
        self,
        *,
        catalog: CatalogArtifact,
        index: IndexArtifact,
        text_embedder: TextEmbedder,
    ) -> None:
        validate_catalog_frame(
            catalog.metadata,
            manifest=catalog.manifest,
            root=catalog.root,
        )
        validate_index_tensors(
            patch_ids=index.patch_ids,
            embeddings=index.embeddings,
            manifest=index.manifest,
            catalog_patch_ids=catalog.patch_ids,
        )
        if index.manifest.catalog_id != catalog.manifest.catalog_id:
            raise ValueError("Index references a different catalog.")
        validate_index_compatibility(index, text_embedder.descriptor)

        self._index = index
        self._text_embedder = text_embedder
        self._metadata_by_patch_id = {
            int(row["patch_id"]): row for row in catalog.metadata.iter_rows(named=True)
        }
        self._catalog_root = catalog.root.resolve()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        normalized_query = _validate_query(query)
        _validate_top_k(top_k)

        query_embedding = self._text_embedder.encode(normalized_query)
        query_vector = _normalized_query_vector(
            query_embedding,
            dimension=self._index.manifest.descriptor.embedding_dimension,
        )
        similarities = self._index.embeddings @ query_vector
        result_count = min(top_k, similarities.shape[0])
        scores, row_indices = torch.topk(
            similarities,
            k=result_count,
            largest=True,
            sorted=True,
        )

        results: list[RetrievalResult] = []
        for rank, (score, row_index) in enumerate(
            zip(scores.tolist(), row_indices.tolist(), strict=True),
            start=1,
        ):
            patch_id = int(self._index.patch_ids[row_index])
            metadata = self._metadata_by_patch_id[patch_id]
            results.append(
                RetrievalResult(
                    rank=rank,
                    patch_id=patch_id,
                    similarity=float(score),
                    description=str(metadata["description"]),
                    source_version=str(metadata["source_version"]),
                    prompt_style=str(metadata["prompt_style"]),
                    wac_image_path=(
                        self._catalog_root / str(metadata["wac_image_path"])
                    ).resolve(),
                    latitude=float(metadata["latitude"]),
                    longitude=float(metadata["longitude"]),
                )
            )
        return results


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty.")
    if len(normalized) > 500:
        raise ValueError("query must contain at most 500 characters.")
    return normalized


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")


def _normalized_query_vector(
    embedding: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    if not isinstance(embedding, torch.Tensor):
        raise TypeError("Text embedder must return a tensor.")
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]
    if embedding.ndim != 1 or embedding.shape[0] != dimension:
        raise ValueError(
            "Text embedding must contain exactly one vector with the index "
            f"dimension ({dimension})."
        )
    vector = embedding.detach().cpu().to(torch.float32)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Text embedding must contain only finite values.")
    norm = torch.linalg.vector_norm(vector)
    if float(norm) == 0:
        raise ValueError("Text embedding must not be the zero vector.")
    return F.normalize(vector, dim=0)
