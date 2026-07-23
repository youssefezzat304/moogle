from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml
from PIL import Image
from rasterio import open as open_raster

from lunar_data.catalog.coordinates import RasterGrid, SimpleCylindricalTransform
from lunar_data.catalog.metadata import (
    CaptionSelection,
    CatalogManifest,
    CatalogRow,
    catalog_rows_to_frame,
    select_display_descriptions,
)
from lunar_data.catalog.validation import validate_catalog_artifact


class RasterPatchSource(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def read_patch(
        self,
        *,
        x_coord: int,
        y_coord: int,
        patch_size: int,
    ) -> Image.Image: ...


class PillowRasterPatchSource:
    """Patch access for small image formats supported by Pillow."""

    def __init__(self, path: str | Path, *, mode: str | None = None) -> None:
        self.path = Path(path)
        self.mode = mode
        self._image = Image.open(self.path)

    @property
    def width(self) -> int:
        return self._image.width

    @property
    def height(self) -> int:
        return self._image.height

    def read_patch(
        self,
        *,
        x_coord: int,
        y_coord: int,
        patch_size: int,
    ) -> Image.Image:
        bounds = (
            x_coord,
            y_coord,
            x_coord + patch_size,
            y_coord + patch_size,
        )
        patch = self._image.crop(bounds)
        return patch.convert(self.mode) if self.mode is not None else patch.copy()

    def close(self) -> None:
        self._image.close()

    def __enter__(self) -> PillowRasterPatchSource:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class RasterioRasterPatchSource:
    """Memory-efficient windowed patch access for large geospatial rasters."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._dataset = open_raster(self.path)

    @property
    def width(self) -> int:
        return self._dataset.width

    @property
    def height(self) -> int:
        return self._dataset.height

    def read_patch(
        self,
        *,
        x_coord: int,
        y_coord: int,
        patch_size: int,
    ) -> Image.Image:
        data = self._dataset.read(
            window=(
                (y_coord, y_coord + patch_size),
                (x_coord, x_coord + patch_size),
            )
        )
        if data.shape[0] == 1:
            image_data = data[0]
        elif data.shape[0] >= 3:
            image_data = data[:3].transpose(1, 2, 0)
        else:
            raise ValueError(
                f"Raster must contain one or at least three bands: {self.path}"
            )
        return Image.fromarray(image_data)

    def close(self) -> None:
        self._dataset.close()

    def __enter__(self) -> RasterioRasterPatchSource:
        return self

    def __exit__(self, *_) -> None:
        self.close()


@dataclass(frozen=True)
class CatalogBuildConfig:
    catalog_id: str
    grid: RasterGrid
    transform: SimpleCylindricalTransform
    caption: CaptionSelection = field(default_factory=CaptionSelection)
    geomap_source: str = "geomap"
    wac_source: str = "wac"
    image_format: str = "WEBP"
    image_extension: str = ".webp"
    image_quality: int = 85
    image_shard_size: int = 1_000

    def __post_init__(self) -> None:
        if not self.catalog_id.strip():
            raise ValueError("catalog_id must be non-empty.")
        if not self.image_extension.startswith("."):
            raise ValueError("image_extension must begin with a dot.")
        if not 1 <= self.image_quality <= 100:
            raise ValueError("image_quality must lie within [1, 100].")
        if self.image_shard_size <= 0:
            raise ValueError("image_shard_size must be positive.")


def build_catalog(
    *,
    captions_path: str | Path,
    geomap_source: RasterPatchSource,
    wac_source: RasterPatchSource,
    output: str | Path,
    config: CatalogBuildConfig,
) -> Path:
    """Build and atomically publish a canonical patch catalog."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"Catalog output already exists: {output_path}")
    _validate_raster_sizes(config.grid, geomap_source, wac_source)

    expected_patch_ids = list(range(config.grid.patch_count))
    descriptions = select_display_descriptions(
        captions_path,
        selection=config.caption,
        expected_patch_ids=expected_patch_ids,
    )
    manifest = CatalogManifest(
        catalog_id=config.catalog_id,
        index_size=config.grid.patch_count,
        grid=config.grid,
        transform=config.transform,
        caption=config.caption,
        geomap_source=config.geomap_source,
        wac_source=config.wac_source,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
        )
    )
    try:
        rows = _write_catalog_rows(
            root=temporary,
            wac_source=wac_source,
            descriptions=descriptions,
            config=config,
        )
        catalog_rows_to_frame(rows).write_parquet(temporary / manifest.metadata_file)
        (temporary / "manifest.yaml").write_text(
            yaml.safe_dump(manifest.to_dict(), sort_keys=False)
        )
        validate_catalog_artifact(temporary)
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def _write_catalog_rows(
    *,
    root: Path,
    wac_source: RasterPatchSource,
    descriptions: dict[int, str],
    config: CatalogBuildConfig,
) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    for patch_id, x_coord, y_coord in config.grid.origins():
        latitude, longitude = config.transform.patch_center(
            x_coord=x_coord,
            y_coord=y_coord,
            patch_size=config.grid.patch_size,
        )
        image_path = _wac_image_path(patch_id, config)
        absolute_image_path = root / image_path
        absolute_image_path.parent.mkdir(parents=True, exist_ok=True)
        patch = wac_source.read_patch(
            x_coord=x_coord,
            y_coord=y_coord,
            patch_size=config.grid.patch_size,
        )
        if patch.size != (config.grid.patch_size, config.grid.patch_size):
            raise ValueError(f"WAC source returned an invalid patch for {patch_id}.")
        save_options = (
            {"quality": config.image_quality}
            if config.image_format.upper() in {"WEBP", "JPEG"}
            else {}
        )
        patch.save(
            absolute_image_path,
            format=config.image_format,
            **save_options,
        )
        patch.close()
        rows.append(
            CatalogRow(
                patch_id=patch_id,
                x_coord=x_coord,
                y_coord=y_coord,
                latitude=latitude,
                longitude=longitude,
                description=descriptions[patch_id],
                source_version=config.caption.source_version,
                prompt_style=config.caption.prompt_style,
                wac_image_path=image_path.as_posix(),
            )
        )
    return rows


def _wac_image_path(patch_id: int, config: CatalogBuildConfig) -> Path:
    shard = patch_id // config.image_shard_size
    filename = f"{patch_id:06d}{config.image_extension}"
    return Path("images") / "wac" / f"{shard:03d}" / filename


def _validate_raster_sizes(
    grid: RasterGrid,
    geomap_source: RasterPatchSource,
    wac_source: RasterPatchSource,
) -> None:
    expected = (grid.width, grid.height)
    if (geomap_source.width, geomap_source.height) != expected:
        raise ValueError("Geomap raster dimensions do not match the canonical grid.")
    if (wac_source.width, wac_source.height) != expected:
        raise ValueError("WAC raster dimensions do not match the canonical grid.")
