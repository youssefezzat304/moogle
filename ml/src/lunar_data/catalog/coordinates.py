from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import degrees, isfinite


LUNAR_RADIUS_METERS = 1_737_400.0


@dataclass(frozen=True)
class RasterGrid:
    """Canonical, column-major patch grid over a raster."""

    width: int
    height: int
    patch_size: int
    stride: int

    def __post_init__(self) -> None:
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("patch_size", self.patch_size),
            ("stride", self.stride),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.patch_size > self.width or self.patch_size > self.height:
            raise ValueError("patch_size must fit within the raster.")

    @property
    def columns(self) -> int:
        return (self.width - self.patch_size) // self.stride + 1

    @property
    def rows(self) -> int:
        return (self.height - self.patch_size) // self.stride + 1

    @property
    def patch_count(self) -> int:
        return self.columns * self.rows

    def origin_for_patch(self, patch_id: int) -> tuple[int, int]:
        if (
            isinstance(patch_id, bool)
            or not isinstance(patch_id, int)
            or not 0 <= patch_id < self.patch_count
        ):
            raise ValueError(f"patch_id must be an integer in [0, {self.patch_count}).")
        column, row = divmod(patch_id, self.rows)
        return column * self.stride, row * self.stride

    def patch_id_for_origin(self, x_coord: int, y_coord: int) -> int:
        if x_coord % self.stride or y_coord % self.stride:
            raise ValueError("Patch origins must be aligned to the configured stride.")
        column = x_coord // self.stride
        row = y_coord // self.stride
        if not 0 <= column < self.columns or not 0 <= row < self.rows:
            raise ValueError("Patch origin lies outside the canonical grid.")
        return column * self.rows + row

    def origins(self) -> Iterator[tuple[int, int, int]]:
        for patch_id in range(self.patch_count):
            yield patch_id, *self.origin_for_patch(patch_id)


@dataclass(frozen=True)
class SimpleCylindricalTransform:
    """North-up simple-cylindrical pixel-to-coordinate transformation."""

    origin_x_meters: float
    origin_y_meters: float
    pixel_width_meters: float
    pixel_height_meters: float
    radius_meters: float = LUNAR_RADIUS_METERS

    def __post_init__(self) -> None:
        values = (
            self.origin_x_meters,
            self.origin_y_meters,
            self.pixel_width_meters,
            self.pixel_height_meters,
            self.radius_meters,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Coordinate transform values must be finite.")
        if self.pixel_width_meters == 0 or self.pixel_height_meters == 0:
            raise ValueError("Pixel dimensions must be non-zero.")
        if self.radius_meters <= 0:
            raise ValueError("radius_meters must be positive.")

    def patch_center(
        self,
        *,
        x_coord: int,
        y_coord: int,
        patch_size: int,
    ) -> tuple[float, float]:
        center_x = x_coord + patch_size / 2
        center_y = y_coord + patch_size / 2
        projected_x = self.origin_x_meters + center_x * self.pixel_width_meters
        projected_y = self.origin_y_meters + center_y * self.pixel_height_meters

        longitude = _normalize_longitude(degrees(projected_x / self.radius_meters))
        latitude = degrees(projected_y / self.radius_meters)
        if not -90 <= latitude <= 90:
            raise ValueError(f"Calculated latitude {latitude} lies outside [-90, 90].")
        return latitude, longitude


def _normalize_longitude(longitude: float) -> float:
    return (longitude + 180.0) % 360.0 - 180.0
