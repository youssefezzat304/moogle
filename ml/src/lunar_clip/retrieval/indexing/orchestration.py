from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch

from lunar_clip.retrieval.indexing.batches import RasterioGeomapBatchSource
from lunar_clip.retrieval.indexing.builder import build_index
from lunar_clip.retrieval.indexing.config import (
    IndexBuildPlan,
    load_index_build_plan,
)
from lunar_clip.retrieval.indexing.contracts import ImageBatch
from lunar_clip.retrieval.indexing.model_loader import load_promoted_embedder
from lunar_clip.retrieval.indexing.validation import (
    validate_index_artifact,
    validate_index_compatibility,
)
from lunar_data.catalog.validation import validate_catalog_artifact


IndexBuildProgress = Callable[[int, int], None]


def build_index_from_recipe(
    path: str | Path,
    *,
    repository_root: str | Path,
    data_root: str | Path,
    progress: IndexBuildProgress | None = None,
) -> Path:
    """Load, validate, and build an embedding index from a YAML recipe."""

    plan = load_index_build_plan(
        path,
        repository_root=repository_root,
        data_root=data_root,
    )
    return build_configured_index(plan, progress=progress)


def build_configured_index(
    plan: IndexBuildPlan,
    *,
    progress: IndexBuildProgress | None = None,
) -> Path:
    """Build and validate a configured production embedding index."""

    _preflight_paths(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production index build.")

    catalog = validate_catalog_artifact(plan.catalog_path)
    batches = RasterioGeomapBatchSource(
        catalog=catalog,
        geomap_path=plan.geomap_path,
        batch_size=plan.batch_size,
    )
    embedder = load_promoted_embedder(
        plan.model_manifest_path,
        device=plan.device,
    )
    if embedder.descriptor.model_id != plan.model_id:
        raise ValueError("Promoted model ID does not match the index recipe.")
    if embedder.descriptor.modality != plan.modality:
        raise ValueError("Promoted model modality does not match the index source.")

    output = build_index(
        catalog=catalog,
        batches=_with_progress(batches, progress),
        embedder=embedder,
        output=plan.output_path,
        config=plan.config,
    )
    artifact = validate_index_artifact(output, catalog=catalog)
    validate_index_compatibility(artifact, embedder.descriptor)
    return output


def _preflight_paths(plan: IndexBuildPlan) -> None:
    for name, path in (
        ("Catalog", plan.catalog_path),
        ("Geomap raster", plan.geomap_path),
        ("Promoted model manifest", plan.model_manifest_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if plan.output_path.exists():
        raise FileExistsError(f"Index output already exists: {plan.output_path}")


def _with_progress(
    batches: Iterable[ImageBatch],
    progress: IndexBuildProgress | None,
) -> Iterator[ImageBatch]:
    total = len(batches) if hasattr(batches, "__len__") else 0
    if progress is not None:
        progress(0, total)
    for completed, batch in enumerate(batches, start=1):
        yield batch
        if progress is not None:
            progress(completed, total)
