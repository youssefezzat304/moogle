from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import rasterio
import torch
from lunar_clip.retrieval.indexing import RasterioGeomapBatchSource
from lunar_data.catalog import (
    CaptionSelection,
    CatalogArtifact,
    CatalogManifest,
    RasterGrid,
    SimpleCylindricalTransform,
)
from rasterio.transform import from_origin


@pytest.fixture
def catalog(tmp_path: Path) -> CatalogArtifact:
    grid = RasterGrid(width=1536, height=512, patch_size=512, stride=512)
    transform = SimpleCylindricalTransform(
        origin_x_meters=-768,
        origin_y_meters=256,
        pixel_width_meters=1,
        pixel_height_meters=-1,
        radius_meters=10_000,
    )
    manifest = CatalogManifest(
        catalog_id="geomap-batch-fixture",
        index_size=grid.patch_count,
        grid=grid,
        transform=transform,
        caption=CaptionSelection(),
        geomap_source="fixture-geomap",
        wac_source="fixture-wac",
    )
    rows = []
    for patch_id, x_coord, y_coord in grid.origins():
        latitude, longitude = transform.patch_center(
            x_coord=x_coord,
            y_coord=y_coord,
            patch_size=grid.patch_size,
        )
        rows.append(
            {
                "patch_id": patch_id,
                "x_coord": x_coord,
                "y_coord": y_coord,
                "latitude": latitude,
                "longitude": longitude,
                "description": f"Description {patch_id}",
                "source_version": "v3.0",
                "prompt_style": "Geologist to Non-Geologist",
                "wac_image_path": f"images/wac/000/{patch_id:06d}.png",
            }
        )
    metadata = pl.DataFrame(
        rows,
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
    )
    return CatalogArtifact(root=tmp_path, manifest=manifest, metadata=metadata)


@pytest.fixture
def geomap_path(tmp_path: Path) -> Path:
    path = tmp_path / "geomap.tif"
    pixels = torch.empty((3, 512, 1536), dtype=torch.uint8)
    for patch_id, x_coord in enumerate((0, 512, 1024)):
        pixels[0, :, x_coord : x_coord + 512] = patch_id + 1
        pixels[1, :, x_coord : x_coord + 512] = patch_id + 11
        pixels[2, :, x_coord : x_coord + 512] = patch_id + 21

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1536,
        height=512,
        count=3,
        dtype="uint8",
        transform=from_origin(0, 512, 1, 1),
    ) as raster:
        raster.write(pixels.numpy())
    return path


def test_geomap_batches_are_ordered_shaped_and_complete(
    catalog: CatalogArtifact,
    geomap_path: Path,
) -> None:
    source = RasterioGeomapBatchSource(
        catalog=catalog,
        geomap_path=geomap_path,
        batch_size=2,
    )

    batches = list(source)

    assert len(source) == 2
    assert [batch.patch_ids.tolist() for batch in batches] == [[0, 1], [2]]
    assert batches[0].images["original"].shape == (2, 3, 512, 512)
    assert batches[1].images["original"].shape == (1, 3, 512, 512)
    assert batches[0].images["original"].dtype == torch.uint8
    assert batches[0].images["original"][0, :, 0, 0].tolist() == [1, 11, 21]
    assert batches[0].images["original"][1, :, 0, 0].tolist() == [2, 12, 22]
    assert batches[1].images["original"][0, :, 0, 0].tolist() == [3, 13, 23]


def test_geomap_batch_source_is_repeatable(
    catalog: CatalogArtifact,
    geomap_path: Path,
) -> None:
    source = RasterioGeomapBatchSource(
        catalog=catalog,
        geomap_path=geomap_path,
        batch_size=2,
    )

    first = list(source)
    second = list(source)

    assert [batch.patch_ids.tolist() for batch in first] == [
        batch.patch_ids.tolist() for batch in second
    ]
    assert all(
        torch.equal(
            first_batch.images["original"],
            second_batch.images["original"],
        )
        for first_batch, second_batch in zip(first, second)
    )


def test_geomap_batch_source_rejects_invalid_batch_size(
    catalog: CatalogArtifact,
    geomap_path: Path,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RasterioGeomapBatchSource(
            catalog=catalog,
            geomap_path=geomap_path,
            batch_size=0,
        )


def test_geomap_batch_source_rejects_incompatible_raster(
    catalog: CatalogArtifact,
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-size.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=512,
        count=3,
        dtype="uint8",
        transform=from_origin(0, 512, 1, 1),
    ):
        pass

    with pytest.raises(ValueError, match="dimensions"):
        RasterioGeomapBatchSource(
            catalog=catalog,
            geomap_path=path,
            batch_size=2,
        )
