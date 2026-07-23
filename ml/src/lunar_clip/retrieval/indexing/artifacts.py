from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from lunar_clip.retrieval.indexing.contracts import EmbeddingDescriptor


INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexManifest:
    index_id: str
    catalog_id: str
    index_size: int
    descriptor: EmbeddingDescriptor
    embeddings_file: str = "embeddings.pt"
    schema_version: int = INDEX_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "index_id": self.index_id,
            "catalog_id": self.catalog_id,
            **self.descriptor.to_dict(),
            "index_size": self.index_size,
            "files": {"embeddings": self.embeddings_file},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexManifest:
        if value.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported index schema_version.")
        try:
            descriptor = EmbeddingDescriptor(
                model_id=str(value["model_id"]),
                checkpoint_sha256=str(value["checkpoint_sha256"]),
                modality=str(value["modality"]),
                preprocessing_id=str(value["preprocessing_id"]),
                embedding_dimension=int(value["embedding_dimension"]),
                embedding_dtype=str(value["embedding_dtype"]),
                normalized=value["normalized"],
                similarity_metric=str(value["similarity_metric"]),
            )
            manifest = cls(
                index_id=str(value["index_id"]),
                catalog_id=str(value["catalog_id"]),
                index_size=int(value["index_size"]),
                descriptor=descriptor,
                embeddings_file=str(value["files"]["embeddings"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid index manifest.") from exc
        if not manifest.index_id.strip() or not manifest.catalog_id.strip():
            raise ValueError("Index and catalog IDs must be non-empty.")
        if manifest.index_size <= 0:
            raise ValueError("index_size must be positive.")
        _validate_relative_path(manifest.embeddings_file)
        return manifest


@dataclass(frozen=True)
class IndexArtifact:
    root: Path
    manifest: IndexManifest
    patch_ids: torch.Tensor
    embeddings: torch.Tensor


def default_index_id(descriptor: EmbeddingDescriptor) -> str:
    serialized = json.dumps(
        descriptor.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    embedding_space_digest = hashlib.sha256(serialized).hexdigest()[:12]
    return f"{descriptor.model_id}-{embedding_space_digest}"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_index_files(
    root: str | Path,
    *,
    manifest: IndexManifest,
    patch_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings.detach().cpu(),
            "patch_ids": patch_ids.detach().cpu(),
        },
        root_path / manifest.embeddings_file,
    )
    (root_path / "manifest.yaml").write_text(
        yaml.safe_dump(manifest.to_dict(), sort_keys=False)
    )


def load_index_artifact(root: str | Path) -> IndexArtifact:
    root_path = Path(root)
    manifest_path = root_path / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    raw_manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw_manifest, dict):
        raise ValueError("Index manifest must contain a mapping.")
    manifest = IndexManifest.from_dict(raw_manifest)

    embeddings_path = root_path / manifest.embeddings_file
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding artifact not found: {embeddings_path}")
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"embeddings", "patch_ids"}:
        raise ValueError("embeddings.pt has an invalid payload.")
    embeddings = payload["embeddings"]
    patch_ids = payload["patch_ids"]
    if not isinstance(embeddings, torch.Tensor) or not isinstance(
        patch_ids, torch.Tensor
    ):
        raise ValueError("embeddings.pt values must be tensors.")
    return IndexArtifact(
        root=root_path,
        manifest=manifest,
        patch_ids=patch_ids,
        embeddings=embeddings,
    )


def _validate_relative_path(value: str) -> None:
    path = Path(value)
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError("Index file paths must be safe relative paths.")
