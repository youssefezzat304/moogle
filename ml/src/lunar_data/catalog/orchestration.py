from __future__ import annotations

from math import isclose
from pathlib import Path

from rasterio import open as open_raster
from rasterio.io import DatasetReader

from lunar_data.catalog.builder import RasterioRasterPatchSource, build_catalog
from lunar_data.catalog.config import (
    CatalogBuildPlan,
    RasterInput,
    load_catalog_build_plan,
)
from lunar_data.catalog.metadata import select_display_descriptions


def preflight_catalog_build(plan: CatalogBuildPlan) -> None:
    """Validate configured catalog inputs without writing an artifact."""

    if not plan.captions_path.is_file():
        raise FileNotFoundError(f"Caption parquet not found: {plan.captions_path}")
    for source in (plan.geomap, plan.wac):
        if not source.path.is_file():
            raise FileNotFoundError(f"Raster source not found: {source.path}")
    if plan.output_path.exists():
        raise FileExistsError(f"Catalog output already exists: {plan.output_path}")

    expected_patch_ids = list(range(plan.config.grid.patch_count))
    select_display_descriptions(
        plan.captions_path,
        selection=plan.config.caption,
        expected_patch_ids=expected_patch_ids,
    )

    with open_raster(plan.geomap.path) as geomap:
        _validate_raster(geomap, plan.geomap, plan=plan)
    with open_raster(plan.wac.path) as wac:
        _validate_raster(wac, plan.wac, plan=plan)
        _validate_wac_georeferencing(wac, plan=plan)


def build_configured_catalog(plan: CatalogBuildPlan) -> Path:
    """Preflight and build a catalog from a loaded production recipe."""

    preflight_catalog_build(plan)
    with (
        RasterioRasterPatchSource(plan.geomap.path) as geomap,
        RasterioRasterPatchSource(plan.wac.path) as wac,
    ):
        return build_catalog(
            captions_path=plan.captions_path,
            geomap_source=geomap,
            wac_source=wac,
            output=plan.output_path,
            config=plan.config,
        )


def build_catalog_from_recipe(
    path: str | Path,
    *,
    repository_root: str | Path,
    data_root: str | Path,
) -> Path:
    """Load, preflight, and build a catalog from a YAML recipe."""

    plan = load_catalog_build_plan(
        path,
        repository_root=repository_root,
        data_root=data_root,
    )
    return build_configured_catalog(plan)


def _validate_raster(
    raster: DatasetReader,
    source: RasterInput,
    *,
    plan: CatalogBuildPlan,
) -> None:
    grid = plan.config.grid
    if (raster.width, raster.height) != (grid.width, grid.height):
        raise ValueError(
            f"{source.source_id} dimensions do not match the canonical grid."
        )
    if raster.count != source.expected_bands:
        raise ValueError(
            f"{source.source_id} has {raster.count} bands; "
            f"expected {source.expected_bands}."
        )
    if tuple(raster.dtypes) != (source.expected_dtype,) * source.expected_bands:
        raise ValueError(
            f"{source.source_id} dtypes do not match {source.expected_dtype}."
        )


def _validate_wac_georeferencing(
    raster: DatasetReader,
    *,
    plan: CatalogBuildPlan,
) -> None:
    if raster.crs is None:
        raise ValueError("WAC raster must define a coordinate reference system.")

    crs = raster.crs.to_dict()
    transform = plan.config.transform
    if crs.get("proj") != "eqc":
        raise ValueError("WAC raster must use an equirectangular projection.")
    if crs.get("units") != "m":
        raise ValueError("WAC raster projection units must be meters.")
    if not _matches(float(crs.get("R", 0.0)), transform.radius_meters):
        raise ValueError("WAC raster lunar radius does not match the catalog recipe.")
    for name in ("lat_ts", "lat_0", "lon_0", "x_0", "y_0"):
        if not _matches(float(crs.get(name, 0.0)), 0.0):
            raise ValueError(f"WAC raster CRS parameter {name} must be zero.")

    affine = raster.transform
    actual = (affine.c, affine.f, affine.a, affine.e, affine.b, affine.d)
    expected = (
        transform.origin_x_meters,
        transform.origin_y_meters,
        transform.pixel_width_meters,
        transform.pixel_height_meters,
        0.0,
        0.0,
    )
    if not all(_matches(left, right) for left, right in zip(actual, expected)):
        raise ValueError(
            "WAC raster affine transform does not match the catalog recipe."
        )


def _matches(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
