from __future__ import annotations

from collections.abc import Iterator
from math import ceil
from pathlib import Path

import torch
from lunar_clip.retrieval.indexing.contracts import ImageBatch
from lunar_data.catalog.metadata import CatalogArtifact
from lunar_data.catalog.validation import validate_catalog_frame
from rasterio import open as open_raster
from rasterio.io import DatasetReader


class RasterioGeomapBatchSource:
    """Yield canonical RGB geomap windows in deterministic catalog order."""

    def __init__(
        self,
        *,
        catalog: CatalogArtifact,
        geomap_path: str | Path,
        batch_size: int,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        self.catalog = catalog
        self.geomap_path = Path(geomap_path)
        self.batch_size = batch_size

        validate_catalog_frame(
            catalog.metadata,
            manifest=catalog.manifest,
            root=catalog.root,
            check_images=False,
        )
        with open_raster(self.geomap_path) as raster:
            self._validate_raster(raster)

    def __len__(self) -> int:
        return ceil(self.catalog.manifest.index_size / self.batch_size)

    def __iter__(self) -> Iterator[ImageBatch]:
        grid = self.catalog.manifest.grid
        rows = self.catalog.metadata.select("patch_id", "x_coord", "y_coord")

        with open_raster(self.geomap_path) as raster:
            self._validate_raster(raster)
            for offset in range(0, rows.height, self.batch_size):
                batch_rows = rows.slice(offset, self.batch_size)
                patch_ids: list[int] = []
                patches: list[torch.Tensor] = []

                for row in batch_rows.iter_rows(named=True):
                    patch_id = int(row["patch_id"])
                    x_coord = int(row["x_coord"])
                    y_coord = int(row["y_coord"])
                    data = raster.read(
                        indexes=(1, 2, 3),
                        window=(
                            (y_coord, y_coord + grid.patch_size),
                            (x_coord, x_coord + grid.patch_size),
                        ),
                    )
                    expected_shape = (3, grid.patch_size, grid.patch_size)
                    if data.shape != expected_shape:
                        raise ValueError(
                            f"Geomap patch {patch_id} has shape {data.shape}; "
                            f"expected {expected_shape}."
                        )
                    patch_ids.append(patch_id)
                    patches.append(torch.from_numpy(data))

                yield ImageBatch(
                    patch_ids=torch.tensor(patch_ids, dtype=torch.int64),
                    images={"original": torch.stack(patches)},
                )

    def _validate_raster(self, raster: DatasetReader) -> None:
        grid = self.catalog.manifest.grid
        if (raster.width, raster.height) != (grid.width, grid.height):
            raise ValueError("Geomap raster dimensions do not match the catalog grid.")
        if raster.count < 3:
            raise ValueError("Geomap raster must contain at least three RGB bands.")
        if tuple(raster.dtypes[:3]) != ("uint8", "uint8", "uint8"):
            raise ValueError("Geomap RGB bands must use uint8 pixels.")
