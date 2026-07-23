from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from lunar_data.catalog.coordinates import RasterGrid, SimpleCylindricalTransform


CATALOG_SCHEMA_VERSION = 1
CATALOG_COLUMNS = (
    "patch_id",
    "x_coord",
    "y_coord",
    "latitude",
    "longitude",
    "description",
    "source_version",
    "prompt_style",
    "wac_image_path",
)


@dataclass(frozen=True)
class CaptionSelection:
    source_version: str = "v3.0"
    prompt_style: str = "Geologist to Non-Geologist"

    def __post_init__(self) -> None:
        if not self.source_version.strip() or not self.prompt_style.strip():
            raise ValueError("Caption selection values must be non-empty.")


@dataclass(frozen=True)
class CatalogRow:
    patch_id: int
    x_coord: int
    y_coord: int
    latitude: float
    longitude: float
    description: str
    source_version: str
    prompt_style: str
    wac_image_path: str


@dataclass(frozen=True)
class CatalogManifest:
    catalog_id: str
    index_size: int
    grid: RasterGrid
    transform: SimpleCylindricalTransform
    caption: CaptionSelection
    geomap_source: str
    wac_source: str
    metadata_file: str = "metadata.parquet"
    wac_images_directory: str = "images/wac"
    schema_version: int = CATALOG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "index_size": self.index_size,
            "grid": {
                "raster_width": self.grid.width,
                "raster_height": self.grid.height,
                "patch_size": self.grid.patch_size,
                "stride": self.grid.stride,
                "columns": self.grid.columns,
                "rows": self.grid.rows,
            },
            "coordinates": {
                "projection": "simple_cylindrical",
                "longitude_convention": "positive_east_[-180,180)",
                "origin_x_meters": self.transform.origin_x_meters,
                "origin_y_meters": self.transform.origin_y_meters,
                "pixel_width_meters": self.transform.pixel_width_meters,
                "pixel_height_meters": self.transform.pixel_height_meters,
                "radius_meters": self.transform.radius_meters,
            },
            "caption": {
                "source_version": self.caption.source_version,
                "prompt_style": self.caption.prompt_style,
            },
            "sources": {
                "geomap": self.geomap_source,
                "wac": self.wac_source,
            },
            "files": {
                "metadata": self.metadata_file,
                "wac_images": self.wac_images_directory,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CatalogManifest:
        if value.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise ValueError("Unsupported catalog schema_version.")
        try:
            grid_value = value["grid"]
            coordinate_value = value["coordinates"]
            caption_value = value["caption"]
            source_value = value["sources"]
            file_value = value["files"]
            grid = RasterGrid(
                width=int(grid_value["raster_width"]),
                height=int(grid_value["raster_height"]),
                patch_size=int(grid_value["patch_size"]),
                stride=int(grid_value["stride"]),
            )
            if (
                int(grid_value["columns"]) != grid.columns
                or int(grid_value["rows"]) != grid.rows
            ):
                raise ValueError("Catalog grid dimensions are inconsistent.")
            if coordinate_value["projection"] != "simple_cylindrical":
                raise ValueError("Unsupported catalog coordinate projection.")
            if coordinate_value["longitude_convention"] != "positive_east_[-180,180)":
                raise ValueError("Unsupported longitude convention.")
            transform = SimpleCylindricalTransform(
                origin_x_meters=float(coordinate_value["origin_x_meters"]),
                origin_y_meters=float(coordinate_value["origin_y_meters"]),
                pixel_width_meters=float(coordinate_value["pixel_width_meters"]),
                pixel_height_meters=float(coordinate_value["pixel_height_meters"]),
                radius_meters=float(coordinate_value["radius_meters"]),
            )
            caption = CaptionSelection(
                source_version=str(caption_value["source_version"]),
                prompt_style=str(caption_value["prompt_style"]),
            )
            manifest = cls(
                catalog_id=str(value["catalog_id"]),
                index_size=int(value["index_size"]),
                grid=grid,
                transform=transform,
                caption=caption,
                geomap_source=str(source_value["geomap"]),
                wac_source=str(source_value["wac"]),
                metadata_file=str(file_value["metadata"]),
                wac_images_directory=str(file_value["wac_images"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid catalog manifest.") from exc
        if manifest.index_size != manifest.grid.patch_count:
            raise ValueError("Catalog index_size does not match its canonical grid.")
        if not manifest.catalog_id.strip():
            raise ValueError("catalog_id must be non-empty.")
        if not manifest.geomap_source.strip() or not manifest.wac_source.strip():
            raise ValueError("Catalog source identifiers must be non-empty.")
        _validate_relative_path(manifest.metadata_file, name="metadata")
        _validate_relative_path(
            manifest.wac_images_directory,
            name="WAC image directory",
        )
        return manifest


@dataclass(frozen=True)
class CatalogArtifact:
    root: Path
    manifest: CatalogManifest
    metadata: pl.DataFrame

    @property
    def patch_ids(self) -> list[int]:
        return self.metadata.get_column("patch_id").to_list()


def select_display_descriptions(
    captions_path: str | Path,
    *,
    selection: CaptionSelection,
    expected_patch_ids: list[int],
) -> dict[int, str]:
    captions = pl.read_parquet(captions_path)
    required = {"patch_id", "source_version", "prompt_style", "text"}
    missing = required - set(captions.columns)
    if missing:
        raise ValueError(
            f"Caption parquet is missing required columns: {sorted(missing)}."
        )

    selected = captions.filter(
        (pl.col("source_version") == selection.source_version)
        & (pl.col("prompt_style") == selection.prompt_style)
    ).select(
        pl.col("patch_id").cast(pl.Int64),
        pl.col("text").cast(pl.Utf8).str.strip_chars().alias("text"),
    )
    duplicate_ids = (
        selected.group_by("patch_id")
        .len()
        .filter(pl.col("len") != 1)
        .get_column("patch_id")
        .to_list()
    )
    if duplicate_ids:
        raise ValueError(
            "Display caption selection must contain exactly one row per patch; "
            f"duplicates found for: {duplicate_ids[:10]}."
        )
    if (
        selected.get_column("text").is_null().any()
        or (selected.get_column("text") == "").any()
    ):
        raise ValueError("Selected display descriptions must be non-empty.")

    descriptions = {int(patch_id): str(text) for patch_id, text in selected.iter_rows()}
    expected = set(expected_patch_ids)
    actual = set(descriptions)
    if actual != expected:
        missing_ids = sorted(expected - actual)
        unexpected_ids = sorted(actual - expected)
        raise ValueError(
            "Display captions do not match the canonical patch IDs. "
            f"Missing: {missing_ids[:10]}; unexpected: {unexpected_ids[:10]}."
        )
    return descriptions


def catalog_rows_to_frame(rows: list[CatalogRow]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "patch_id": row.patch_id,
                "x_coord": row.x_coord,
                "y_coord": row.y_coord,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "description": row.description,
                "source_version": row.source_version,
                "prompt_style": row.prompt_style,
                "wac_image_path": row.wac_image_path,
            }
            for row in rows
        ],
        schema={
            "patch_id": pl.Int64,
            "x_coord": pl.Int64,
            "y_coord": pl.Int64,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "description": pl.String,
            "source_version": pl.String,
            "prompt_style": pl.String,
            "wac_image_path": pl.String,
        },
    ).select(CATALOG_COLUMNS)


def load_catalog_artifact(root: str | Path) -> CatalogArtifact:
    root_path = Path(root)
    manifest_path = root_path / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Catalog manifest not found: {manifest_path}")
    raw_manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw_manifest, dict):
        raise ValueError("Catalog manifest must contain a mapping.")
    manifest = CatalogManifest.from_dict(raw_manifest)
    metadata_path = root_path / manifest.metadata_file
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Catalog metadata not found: {metadata_path}")
    return CatalogArtifact(
        root=root_path,
        manifest=manifest,
        metadata=pl.read_parquet(metadata_path),
    )


def _validate_relative_path(value: str, *, name: str) -> None:
    path = Path(value)
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Catalog {name} path must be a safe relative path.")
