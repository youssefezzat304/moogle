from __future__ import annotations

from math import isclose
from pathlib import Path

import polars as pl

from lunar_data.catalog.metadata import (
    CATALOG_COLUMNS,
    CatalogArtifact,
    CatalogManifest,
    load_catalog_artifact,
)


def validate_catalog_frame(
    metadata: pl.DataFrame,
    *,
    manifest: CatalogManifest,
    root: Path,
    check_images: bool = True,
) -> None:
    if tuple(metadata.columns) != CATALOG_COLUMNS:
        raise ValueError(
            f"Catalog metadata columns must be {CATALOG_COLUMNS}; "
            f"found {tuple(metadata.columns)}."
        )
    if metadata.height != manifest.index_size:
        raise ValueError("Catalog metadata row count does not match index_size.")

    patch_ids = metadata.get_column("patch_id").to_list()
    expected_patch_ids = list(range(manifest.grid.patch_count))
    if patch_ids != expected_patch_ids:
        raise ValueError("Catalog patch IDs must be unique, contiguous, and ordered.")

    for row in metadata.iter_rows(named=True):
        patch_id = int(row["patch_id"])
        expected_x, expected_y = manifest.grid.origin_for_patch(patch_id)
        if (int(row["x_coord"]), int(row["y_coord"])) != (
            expected_x,
            expected_y,
        ):
            raise ValueError(f"Patch {patch_id} does not use canonical coordinates.")

        expected_latitude, expected_longitude = manifest.transform.patch_center(
            x_coord=expected_x,
            y_coord=expected_y,
            patch_size=manifest.grid.patch_size,
        )
        if not isclose(
            float(row["latitude"]), expected_latitude, abs_tol=1e-9
        ) or not isclose(float(row["longitude"]), expected_longitude, abs_tol=1e-9):
            raise ValueError(
                f"Patch {patch_id} has inconsistent geographic coordinates."
            )
        if not -90 <= float(row["latitude"]) <= 90:
            raise ValueError(f"Patch {patch_id} has invalid latitude.")
        if not -180 <= float(row["longitude"]) < 180:
            raise ValueError(f"Patch {patch_id} has invalid longitude.")

        if (
            str(row["source_version"]) != manifest.caption.source_version
            or str(row["prompt_style"]) != manifest.caption.prompt_style
        ):
            raise ValueError(f"Patch {patch_id} has inconsistent caption provenance.")
        if not str(row["description"]).strip():
            raise ValueError(f"Patch {patch_id} has an empty description.")

        image_path = Path(str(row["wac_image_path"]))
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError(f"Patch {patch_id} has an unsafe WAC image path.")
        if not image_path.is_relative_to(Path(manifest.wac_images_directory)):
            raise ValueError(
                f"Patch {patch_id} WAC image lies outside the configured directory."
            )
        if check_images and not (root / image_path).is_file():
            raise ValueError(f"Patch {patch_id} WAC image does not exist.")


def validate_catalog_artifact(
    root: str | Path,
    *,
    check_images: bool = True,
) -> CatalogArtifact:
    artifact = load_catalog_artifact(root)
    validate_catalog_frame(
        artifact.metadata,
        manifest=artifact.manifest,
        root=artifact.root,
        check_images=check_images,
    )
    return artifact
