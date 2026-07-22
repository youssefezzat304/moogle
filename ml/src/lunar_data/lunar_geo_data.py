"""Lunar Geological Dataset for PyTorch.

This module provides a PyTorch Dataset that pairs lunar Wide Angle Camera (WAC)
imagery with geological segmentation maps. The full-resolution global maps are
loaded once into memory, and individual training patches are extracted on the fly
using a regular grid defined by ``patch_size`` and ``stride``.

Typical usage::

    from lunarGeoData import LunarGeoData

    dataset = LunarGeoData(root="/path/to/lunar-data", patch_size=256, stride=128)
    sample  = dataset[0]

    wac_patch    = sample["wac"]["tensor"]      # shape: (1, 256, 256), float32 in [0, 1]
    geomap_patch = sample["geomap"]["tensor"]   # shape: (num_classes, 256, 256), float32 one-hot
    geomap_rgb   = sample["geomap"]["original"] # shape: (3, 256, 256), original RGB
"""

import os
import json
from typing import Callable, Dict, List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError for large lunar mosaics


# ---------------------------------------------------------------------------
# Standalone loaders – called once at dataset construction time
# ---------------------------------------------------------------------------
def load_geomap(root: str, folder: str = "UnifiedGeoMap") -> torch.Tensor:
    """Load the full-resolution geological map as an integer RGB tensor.

    Args:
        root: Root directory that contains the data folders.
        folder: Subfolder with the geomap TIFF file.

    Returns:
        Integer tensor of shape ``(3, H, W)`` with values in ``[0, 255]``.
    """
    path = os.path.join(root, folder, "geomap_map_zoom6_resized.tiff")
    image = Image.open(path)
    return pil_to_tensor(image)  # uint8 -> int tensor [C, H, W]

def load_wac(root: str, folder: str = "wac") -> torch.Tensor:
    """Load the Wide Angle Camera (WAC) mosaic as a normalised float tensor.

    Args:
        root: Root directory that contains the data folders.
        folder: Subfolder with the WAC TIFF file.

    Returns:
        Float tensor of shape ``(1, H, W)`` with values in ``[0.0, 1.0]``.
    """
    path = os.path.join(
        root, folder, "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    )
    image = Image.open(path)
    return pil_to_tensor(image).float() / 255.0


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a uint8 tensor shaped (C, H, W)."""
    if image.mode == "1":
        image = image.convert("L")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    channels = len(image.getbands())
    return data.view(image.height, image.width, channels).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LunarGeoData(Dataset):
    """PyTorch Dataset that serves paired WAC / geological-map patches.

    On construction the class:
    1. Loads the full WAC and geomap images into memory.
    2. Converts the RGB geomap into a **one-hot segmentation tensor** with one channel per geological class (done once - not repeated every iteration).
    3. Builds a regular grid of ``(x, y)`` extraction points spaced by ``stride`` pixels.

    Each ``__getitem__`` call then returns a lightweight tensor slice - no per-sample colour matching or post-processing is needed during training.

    Attributes:
        wac: Full WAC tensor, shape ``(1, H, W)``, float32.
        geomap_class_map: Pre-computed class-index map, shape ``(H, W)``, uint8.
            Each pixel holds the legend index of its geological class.
        geomap_rgb: Original RGB geomap tensor kept for visualisation, shape ``(3, H, W)``.
        num_classes: Number of geological classes (channels in one-hot output).
        grid_points: Tensor of ``(x, y)`` patch origins, shape ``(N, 2)``.
        legend: Mapping from class abbreviation to metadata dict containing
            - ``"color"`` (hex), ``"rgb"`` (list), ``"description"`` (str) and
            - ``"index"`` (int channel index in the one-hot output).
    """
    def __init__(self, root: str, patch_size: int = 256, stride: int = 128, transform: Dict[str, Callable] | None = None):
        """Create a new LunarGeoData dataset.

        Args:
            root: Path to the directory that contains ``UnifiedGeoMap/`` and ``wac/`` sub-folders.
            patch_size: Height and width (in pixels) of each extracted patch.
            stride: Step size (in pixels) between neighbouring patch origins. Use ``stride < patch_size`` for overlapping patches.
            transform: Optional dict of per-modality transforms, e.g. ``{"wac": some_augmentation}``.  Each value is a callable that receives and returns a tensor.
        """
        self.root = root
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # --- Load raw data ------------------------------------------------
        self.wac: torch.Tensor = load_wac(root)
        geomap_rgb: torch.Tensor = load_geomap(root)

        # --- Build legend -------------------------------------------------
        self.legend = self._load_legend()

        # --- Pre-compute class-index map (stores one uint8 per pixel) -----
        self.geomap_rgb = geomap_rgb
        self.num_classes = len(self.legend)
        self.geomap_class_map = self._precompute_class_map(geomap_rgb)

        # --- Build extraction grid ----------------------------------------
        self.grid_points = self._build_grid()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _load_legend(self) -> Dict[str, Dict[str, Union[str, List[int], int]]]:
            """Read the local geomap legend and enrich it with RGB values."""
            
            # 1. Dynamically find the directory where this script (lunarGeoData.py) lives
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. Point to the legend.json file sitting right next to it
            legend_path = os.path.join(current_dir, "legend.json")
            
            # 3. Read the JSON file
            with open(legend_path, "r") as f:
                raw_legend = json.load(f)

            # 4. Process the RGB values just like before
            legend = {
                abbrev: {
                    **meta,
                    "rgb": _hex_to_rgb(str(meta.get("color", ""))),
                }
                for abbrev, meta in raw_legend.items()
            }
            
            # 5. Append the Background fallback class
            legend["Bg"] = {
                "color": "#000000",
                "rgb": [0, 0, 0],
                "description": "Background or line markings",
                "long_description": "Background or line markings", # Prevents KeyError in utils.py
            }
            return legend

    def _precompute_class_map(self, geomap_rgb: torch.Tensor) -> torch.Tensor:
        """Convert the RGB geomap into a single-channel class-index map.

        Each pixel is assigned the index of its matching legend colour.
        Storing one ``uint8`` per pixel instead of a full one-hot volume
        keeps memory usage at ``H * W`` bytes (vs ``num_classes * H * W * 4``).
        The one-hot expansion happens in ``__getitem__`` on the small patch.

        Args:
            geomap_rgb: Integer RGB tensor, shape ``(3, H, W)``.

        Returns:
            ``uint8`` tensor of shape ``(H, W)`` with class indices.
        """
        _, h, w = geomap_rgb.shape
        class_map = torch.zeros((h, w), dtype=torch.uint8)

        for i, (abbrev, meta) in enumerate(self.legend.items()):
            color = torch.tensor(meta["rgb"], dtype=geomap_rgb.dtype).view(3, 1, 1)
            mask = (geomap_rgb == color).all(dim=0)
            class_map[mask] = i
            self.legend[abbrev]["index"] = i

        return class_map

    def _build_grid(self) -> torch.Tensor:
        """Create the regular grid of patch extraction origins.

        The grid is derived from the WAC dimensions so that every patch fits
        entirely within the image boundaries.

        Returns:
            Tensor of shape ``(N, 2)`` where each row is an ``(x, y)`` origin.
        """
        _, h, w = self.wac.shape

        xs = torch.arange(0, w - self.patch_size + 1, self.stride)
        ys = torch.arange(0, h - self.patch_size + 1, self.stride)

        grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1)
        return grid.reshape(-1, 2)  # (num_patches, 2)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the total number of patches in the dataset."""
        return len(self.grid_points)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a single sample as a nested dict of modalities.

        The nested structure ``{modality: {field: value}}`` makes it easy to attach extra fields per modality (e.g. the original RGB geomap, or a
        text caption) without breaking downstream code that indexes by
        modality name first.

        Args:
            idx: Patch index (``0 <= idx < len(self)``).

        Returns:
            Dict of modalities, each containing a sub-dict::

                {
                    "wac":    {"tensor": Tensor(1, H, W)},
                    "geomap": {"tensor": Tensor(num_classes, H, W),
                               "original": Tensor(3, H, W)},
                }
        """
        x, y = int(self.grid_points[idx, 0]), int(self.grid_points[idx, 1])
        s = self.patch_size

        # Expand the small patch from class indices to one-hot on the fly
        class_patch = self.geomap_class_map[y : y + s, x : x + s].long()  # (H, W)
        one_hot = torch.zeros(
            (self.num_classes, s, s), dtype=torch.float32
        )
        one_hot.scatter_(0, class_patch.unsqueeze(0), 1.0)

        output: Dict[str, Dict[str, torch.Tensor]] = {
            "wac": {
                "tensor": self.wac[:, y : y + s, x : x + s],
            },
            "geomap": {
                "tensor": one_hot,
                "original": self.geomap_rgb[:, y : y + s, x : x + s],
            },
        }

        if self.transform:
            for key, fn in self.transform.items():
                if key in output:
                    output[key]["tensor"] = fn(output[key]["tensor"])

        return output

    # ------------------------------------------------------------------
    # Visualisation / analysis helpers
    # ------------------------------------------------------------------
    def get_rgb_patch(self, idx: int) -> torch.Tensor:
        """Return the original RGB geomap patch (useful for visualisation).

        Args:
            idx: Patch index.

        Returns:
            Integer tensor of shape ``(3, patch_size, patch_size)``.
        """
        x, y = int(self.grid_points[idx, 0]), int(self.grid_points[idx, 1])
        s = self.patch_size
        return self.geomap_rgb[:, y : y + s, x : x + s]

    def identify_classes_in_patch(self, idx: int) -> Tuple[List[str], List[str]]:
        """List the geological classes present in a given patch.

        Args:
            idx: Patch index.

        Returns:
            Tuple of ``(hex_colors, abbreviations)`` for every class that has
            at least one pixel in the patch.
        """
        patch = self.get_rgb_patch(idx)
        return identify_classes_in_image(patch, self.legend)


# ---------------------------------------------------------------------------
# Pure-function utilities (no ``self`` needed)
# ---------------------------------------------------------------------------
def _hex_to_rgb(hex_color: str) -> List[int]:
    """Convert a hex colour string like ``'#FCDC0A'`` to ``[R, G, B]``."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]

def identify_classes_in_image(image: torch.Tensor, legend: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
    """Identify which geological classes appear in an RGB image tensor.

    For each pixel the nearest legend colour (by squared Euclidean distance)
    is found and the corresponding class is recorded.

    Args:
        image: Integer tensor of shape ``(3, H, W)``.
        legend: The ``legend`` dict from a :class:`LunarGeoData` instance.

    Returns:
        Tuple of ``(hex_colors, abbreviations)`` for every unique class found.
    """
    color_to_abbrev = {v["color"]: k for k, v in legend.items()}
    colors_hex = list(color_to_abbrev.keys())
    colors_rgb = torch.tensor(
        [_hex_to_rgb(c) for c in colors_hex], dtype=torch.int64
    )  # (num_colors, 3)

    pixels = image.permute(1, 2, 0).reshape(-1, 3).to(torch.int64)  # (H*W, 3)
    dists = ((pixels[:, None, :] - colors_rgb[None, :, :]) ** 2).sum(dim=-1)
    closest = torch.argmin(dists, dim=-1)  # (H*W,)

    unique_indices = closest.unique().tolist()
    found_hex = [colors_hex[i] for i in unique_indices]
    found_abbrev = [color_to_abbrev[h] for h in found_hex]
    return found_hex, found_abbrev

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = "/home/pg2026/data"

    dataset = LunarGeoData(root=root, patch_size=256, stride=128)

    print(f"Total number of patches: {len(dataset)}")
    sample = dataset[0]
    print(
        f"WAC shape: {sample['wac']['tensor'].shape}, "
        f"Geomap shape: {sample['geomap']['tensor'].shape}"
    )
