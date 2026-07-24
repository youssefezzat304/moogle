from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lunar_clip.retrieval.indexing.builder import IndexBuildConfig


INDEX_BUILD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexBuildPlan:
    config: IndexBuildConfig
    catalog_path: Path
    geomap_path: Path
    model_id: str
    model_manifest_path: Path
    modality: str
    device: str
    batch_size: int
    output_path: Path


def load_index_build_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
    data_root: str | Path,
) -> IndexBuildPlan:
    """Load a strict production embedding-index recipe."""

    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Index build recipe not found: {recipe_path}")
    try:
        raw = yaml.safe_load(recipe_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError("Index build recipe is not valid YAML.") from exc

    root = _mapping(raw, "index recipe")
    _exact_keys(
        root,
        {
            "schema_version",
            "index_id",
            "catalog",
            "source",
            "model",
            "build",
            "output",
        },
        "index recipe",
    )
    schema_version = _integer(root["schema_version"], "schema_version")
    if schema_version != INDEX_BUILD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported index build schema_version: {schema_version}.")

    catalog_path = _single_path(root["catalog"], "catalog")
    source_path, modality = _load_source(root["source"])
    model_manifest_path, model_id, device = _load_model(root["model"])
    batch_size = _load_build(root["build"])
    output_path = _single_path(root["output"], "output")

    repository = Path(repository_root).resolve()
    data = Path(data_root).resolve()
    return IndexBuildPlan(
        config=IndexBuildConfig(
            index_id=_string(root["index_id"], "index_id"),
        ),
        catalog_path=_resolve_relative_path(
            catalog_path,
            root=repository,
            name="catalog.path",
        ),
        geomap_path=_resolve_relative_path(
            source_path,
            root=data,
            name="source.path",
        ),
        model_id=model_id,
        model_manifest_path=_resolve_relative_path(
            model_manifest_path,
            root=repository,
            name="model.manifest",
        ),
        modality=modality,
        device=device,
        batch_size=batch_size,
        output_path=_resolve_relative_path(
            output_path,
            root=repository,
            name="output.path",
        ),
    )


def _single_path(value: Any, name: str) -> Path:
    path_value = _mapping(value, name)
    _exact_keys(path_value, {"path"}, name)
    return _relative_path(path_value["path"], f"{name}.path")


def _load_source(value: Any) -> tuple[Path, str]:
    source_value = _mapping(value, "source")
    _exact_keys(source_value, {"modality", "path"}, "source")
    modality = _string(source_value["modality"], "source.modality")
    if modality != "geomap":
        raise ValueError("Index build recipes currently support geomap sources.")
    return _relative_path(source_value["path"], "source.path"), modality


def _load_model(value: Any) -> tuple[Path, str, str]:
    model_value = _mapping(value, "model")
    _exact_keys(model_value, {"model_id", "manifest", "device"}, "model")
    device = _string(model_value["device"], "model.device")
    if device != "cuda":
        raise ValueError("Production index builds require model.device: cuda.")
    return (
        _relative_path(model_value["manifest"], "model.manifest"),
        _string(model_value["model_id"], "model.model_id"),
        device,
    )


def _load_build(value: Any) -> int:
    build_value = _mapping(value, "build")
    _exact_keys(build_value, {"batch_size"}, "build")
    batch_size = _integer(build_value["batch_size"], "build.batch_size")
    if batch_size <= 0:
        raise ValueError("build.batch_size must be positive.")
    return batch_size


def _resolve_relative_path(value: Path, *, root: Path, name: str) -> Path:
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} must resolve within its configured root.")
    return resolved


def _relative_path(value: Any, name: str) -> Path:
    path = Path(_string(value, name))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe relative path.")
    return path


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a mapping with string keys.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{name} fields are invalid. Missing: {missing}; unknown: {unknown}."
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value
