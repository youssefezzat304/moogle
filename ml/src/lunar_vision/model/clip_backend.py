"""clip_backend.py — shared contract for vision encoder backends.

Mirrors lunar_text.model.clip_backend.TextEncoderBackend: a thin nn.Module
base class declaring the contract lunar_clip.encoders.vision.
lunar_vision_encoder.LunarVisionEncoder expects from a vision backend, so
CLIP training can stay backend-agnostic.

Unlike the text backend, this does not provide a default encode_retrieval()
template — vision backends differ in their retrieval mechanics (e.g. Fusion
rejects non-native resolutions, Geo and WAC resize their positional
embeddings), so each backend implements encode_retrieval() fully on its own.

resize_pos_embed()/source_pos_grid_size() below are the one piece of
mechanics genuinely shared between backends that resize (Geo's
lunar_vision.model.geo.model.Encoder and WAC's
lunar_vision.model.wac.encoder.WACEncoder) — factored out here so that math
isn't duplicated, not because every backend is expected to use it (Fusion
deliberately doesn't).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class VisionEncoderBackend(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def encode_retrieval(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        """Return one retrieval vector per image."""
        raise NotImplementedError


def resize_pos_embed(
    pos_embed: torch.Tensor,
    source_grid_size: tuple[int, int],
    target_grid_size: tuple[int, int],
) -> torch.Tensor:
    """Bicubically resize a (1, num_patches, hidden_dim) positional embedding
    from source_grid_size to target_grid_size. No-op if they already match."""
    if source_grid_size == target_grid_size:
        return pos_embed

    hidden_dim = pos_embed.shape[-1]
    pos_embed = pos_embed.reshape(1, source_grid_size[0], source_grid_size[1], hidden_dim)
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=target_grid_size, mode="bicubic", align_corners=False,
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1)
    return pos_embed.reshape(1, target_grid_size[0] * target_grid_size[1], -1)


def source_pos_grid_size(token_count: int, image_size: int, patch_size: int) -> tuple[int, int]:
    """Recover the square (grid_h, grid_w) a positional embedding with
    token_count entries was trained at, preferring image_size/patch_size and
    falling back to sqrt(token_count) if that doesn't divide evenly."""
    source_side = image_size // patch_size if image_size > 0 else 0
    if source_side * source_side == token_count:
        return source_side, source_side

    source_side = int(token_count**0.5)
    if source_side * source_side != token_count:
        raise ValueError(
            f"Positional embeddings must form a square grid. Got {token_count} positions."
        )
    return source_side, source_side
