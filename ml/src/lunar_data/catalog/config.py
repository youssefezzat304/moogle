from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from lunar_data.catalog.builder import CatalogBuildConfig
from lunar_data.catalog.coordinates import RasterGrid, SimpleCylindricalTransform
from lunar_data.catalog.metadata import CaptionSelection


CATALOG_BUILD_SCHEMA_VERSION = 1
CATALOG_GRID_ORDERING = "column_major"


@dataclass(frozen=True)
class RasterInput:
    source_id: str
    path: Path
    expected_bands: int
    expected_dtype: str


@dataclass(frozen=True)
class CatalogBuildPlan:
    config: CatalogBuildConfig
    captions_path: Path
    geomap: RasterInput
    wac: RasterInput
    output_path: Path


def load_catalog_build_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
    data_root: str | Path,
) -> CatalogBuildPlan:
    """Load a strict production catalog recipe into typed build inputs."""

    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Catalog build recipe not found: {recipe_path}")
    try:
        raw = yaml.safe_load(recipe_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError("Catalog build recipe is not valid YAML.") from exc

    root = _mapping(raw, "catalog recipe")
    _exact_keys(
        root,
        {
            "schema_version",
            "catalog_id",
            "grid",
            "coordinates",
            "sources",
            "captions",
            "images",
            "output",
        },
        "catalog recipe",
    )
    schema_version = _integer(root["schema_version"], "schema_version")
    if schema_version != CATALOG_BUILD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported catalog build schema_version: {schema_version}.")

    grid = _load_grid(root["grid"])
    transform = _load_transform(root["coordinates"])
    geomap_value, wac_value = _load_sources(root["sources"])
    caption, captions_path_value = _load_caption(root["captions"])
    image_values = _load_images(root["images"])
    output_path_value = _load_output(root["output"])

    repository = Path(repository_root).resolve()
    data = Path(data_root).resolve()
    geomap = _resolve_raster_input(geomap_value, root=data, name="sources.geomap")
    wac = _resolve_raster_input(wac_value, root=data, name="sources.wac")
    captions_path = _resolve_relative_path(
        captions_path_value,
        root=repository,
        name="captions.path",
    )
    output_path = _resolve_relative_path(
        output_path_value,
        root=repository,
        name="output.path",
    )

    config = CatalogBuildConfig(
        catalog_id=_string(root["catalog_id"], "catalog_id"),
        grid=grid,
        transform=transform,
        caption=caption,
        geomap_source=geomap.source_id,
        wac_source=wac.source_id,
        image_format=image_values["format"],
        image_extension=image_values["extension"],
        image_quality=image_values["quality"],
        image_shard_size=image_values["shard_size"],
    )
    return CatalogBuildPlan(
        config=config,
        captions_path=captions_path,
        geomap=geomap,
        wac=wac,
        output_path=output_path,
    )


def _load_grid(value: Any) -> RasterGrid:
    grid_value = _mapping(value, "grid")
    _exact_keys(
        grid_value,
        {
            "raster_width",
            "raster_height",
            "patch_size",
            "stride",
            "columns",
            "rows",
            "patch_count",
            "ordering",
        },
        "grid",
    )
    ordering = _string(grid_value["ordering"], "grid.ordering")
    if ordering != CATALOG_GRID_ORDERING:
        raise ValueError(f"Unsupported catalog grid ordering: {ordering}.")

    grid = RasterGrid(
        width=_integer(grid_value["raster_width"], "grid.raster_width"),
        height=_integer(grid_value["raster_height"], "grid.raster_height"),
        patch_size=_integer(grid_value["patch_size"], "grid.patch_size"),
        stride=_integer(grid_value["stride"], "grid.stride"),
    )
    declared = {
        "columns": _integer(grid_value["columns"], "grid.columns"),
        "rows": _integer(grid_value["rows"], "grid.rows"),
        "patch_count": _integer(grid_value["patch_count"], "grid.patch_count"),
    }
    calculated = {
        "columns": grid.columns,
        "rows": grid.rows,
        "patch_count": grid.patch_count,
    }
    for name, expected in calculated.items():
        if declared[name] != expected:
            raise ValueError(
                f"grid.{name} is {declared[name]}; calculated value is {expected}."
            )
    return grid


def _load_transform(value: Any) -> SimpleCylindricalTransform:
    coordinate_value = _mapping(value, "coordinates")
    _exact_keys(
        coordinate_value,
        {
            "projection",
            "longitude_convention",
            "origin_x_meters",
            "origin_y_meters",
            "pixel_width_meters",
            "pixel_height_meters",
            "radius_meters",
        },
        "coordinates",
    )
    projection = _string(coordinate_value["projection"], "coordinates.projection")
    if projection != "simple_cylindrical":
        raise ValueError(f"Unsupported catalog coordinate projection: {projection}.")
    longitude_convention = _string(
        coordinate_value["longitude_convention"],
        "coordinates.longitude_convention",
    )
    if longitude_convention != "positive_east_[-180,180)":
        raise ValueError(
            f"Unsupported longitude convention: {longitude_convention}."
        )
    return SimpleCylindricalTransform(
        origin_x_meters=_number(
            coordinate_value["origin_x_meters"],
            "coordinates.origin_x_meters",
        ),
        origin_y_meters=_number(
            coordinate_value["origin_y_meters"],
            "coordinates.origin_y_meters",
        ),
        pixel_width_meters=_number(
            coordinate_value["pixel_width_meters"],
            "coordinates.pixel_width_meters",
        ),
        pixel_height_meters=_number(
            coordinate_value["pixel_height_meters"],
            "coordinates.pixel_height_meters",
        ),
        radius_meters=_number(
            coordinate_value["radius_meters"],
            "coordinates.radius_meters",
        ),
    )


def _load_sources(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sources_value = _mapping(value, "sources")
    _exact_keys(sources_value, {"geomap", "wac"}, "sources")
    return (
        _load_raster_source(sources_value["geomap"], "sources.geomap"),
        _load_raster_source(sources_value["wac"], "sources.wac"),
    )


def _load_raster_source(value: Any, name: str) -> dict[str, Any]:
    source_value = _mapping(value, name)
    _exact_keys(
        source_value,
        {"id", "path", "expected_bands", "expected_dtype"},
        name,
    )
    return {
        "source_id": _string(source_value["id"], f"{name}.id"),
        "path": _relative_path(source_value["path"], f"{name}.path"),
        "expected_bands": _positive_integer(
            source_value["expected_bands"],
            f"{name}.expected_bands",
        ),
        "expected_dtype": _string(
            source_value["expected_dtype"],
            f"{name}.expected_dtype",
        ),
    }


def _load_caption(value: Any) -> tuple[CaptionSelection, Path]:
    caption_value = _mapping(value, "captions")
    _exact_keys(
        caption_value,
        {"path", "source_version", "prompt_style"},
        "captions",
    )
    return (
        CaptionSelection(
            source_version=_string(
                caption_value["source_version"],
                "captions.source_version",
            ),
            prompt_style=_string(
                caption_value["prompt_style"],
                "captions.prompt_style",
            ),
        ),
        _relative_path(caption_value["path"], "captions.path"),
    )


def _load_images(value: Any) -> dict[str, Any]:
    image_value = _mapping(value, "images")
    _exact_keys(
        image_value,
        {"format", "extension", "quality", "shard_size"},
        "images",
    )
    image_format = _string(image_value["format"], "images.format").upper()
    if image_format != "WEBP":
        raise ValueError("Catalog build recipes currently support WEBP output.")
    extension = _string(image_value["extension"], "images.extension").lower()
    if extension != ".webp":
        raise ValueError("images.extension must be .webp for WEBP output.")
    return {
        "format": image_format,
        "extension": extension,
        "quality": _integer(image_value["quality"], "images.quality"),
        "shard_size": _positive_integer(
            image_value["shard_size"],
            "images.shard_size",
        ),
    }


def _load_output(value: Any) -> Path:
    output_value = _mapping(value, "output")
    _exact_keys(output_value, {"path"}, "output")
    return _relative_path(output_value["path"], "output.path")


def _resolve_raster_input(
    value: dict[str, Any],
    *,
    root: Path,
    name: str,
) -> RasterInput:
    return RasterInput(
        source_id=value["source_id"],
        path=_resolve_relative_path(value["path"], root=root, name=f"{name}.path"),
        expected_bands=value["expected_bands"],
        expected_dtype=value["expected_dtype"],
    )


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


def _positive_integer(value: Any, name: str) -> int:
    integer = _integer(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive.")
    return integer


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number
