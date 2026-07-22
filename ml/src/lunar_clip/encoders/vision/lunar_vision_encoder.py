from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lunar_clip.contracts.batches import RetrievalBatch
from lunar_vision.model.clip_backend import VisionEncoderBackend


class LunarVisionEncoder(nn.Module):
    """Single CLIP-owned vision encoder for the fixed LunarCLIP vision backends."""

    _MODALITIES = {
        "geo": "geomap",
        "wac": "wac",
        "fusion": "wac",
    }
    _TENSOR_KEYS = {
        "geo": ("original", "tensor"),
        "wac": ("tensor",),
        "fusion": ("tensor",),
    }

    def __init__(
        self,
        encoder: str,
        checkpoint_path: str | None = None,
        freeze_encoder: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder.lower()
        if self.encoder not in self._MODALITIES:
            raise ValueError(f"Unknown vision encoder '{encoder}'. Available encoders: {sorted(self._MODALITIES)}")
        self.expected_modality = self._MODALITIES[self.encoder]

        if self.encoder == "geo":
            self.model, self.output_dim = self._load_geo(checkpoint_path)
        elif self.encoder == "wac":
            self.model, self.output_dim = self._load_wac(checkpoint_path)
        elif self.encoder == "fusion":
            self.model, self.output_dim = self._load_fusion(checkpoint_path)

        self.freeze_encoder = freeze_encoder
        # TODO: Replace this parameter-name-based freeze policy with a backend-owned
        # freeze contract. Backends should define which retrieval parameters remain
        # trainable when their pretrained encoder is frozen.
        if freeze_encoder:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = name.endswith("retrieval")
        if device is not None:
            self.to(torch.device(device))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.model.eval()
        return self

    def encode_image(
        self,
        batch: torch.Tensor | dict[str, Any],
        modality: str,
    ) -> RetrievalBatch:
        if modality != self.expected_modality:
            raise ValueError(
                f"LunarVisionEncoder({self.encoder}) only supports "
                f"modality={self.expected_modality!r}. Got {modality!r}."
            )
        vectors = self._encode_image(batch)
        return RetrievalBatch(vectors=vectors.float(), metadata={"modality": modality})

    def _load_geo(self, checkpoint_path: str | None) -> tuple[VisionEncoderBackend, int]:
        if checkpoint_path is None:
            raise ValueError("Geo vision encoder requires checkpoint_path.")

        from lunar_vision.model.geo.encoder import GeoEncoder

        model = GeoEncoder.load_from_checkpoint(checkpoint_path)
        return model, model.output_dim

    def _load_wac(self, checkpoint_path: str | None) -> tuple[VisionEncoderBackend, int]:
        if checkpoint_path is None:
            raise ValueError("WAC vision encoder requires checkpoint_path.")

        from lunar_vision.model.wac.encoder import WACEncoder

        model = WACEncoder.load_from_checkpoint(checkpoint_path)
        return model, model.output_dim

    def _load_fusion(self, checkpoint_path: str | None) -> tuple[VisionEncoderBackend, int]:
        if checkpoint_path is None:
            raise ValueError("Fusion vision encoder requires checkpoint_path.")

        from lunar_vision.model.fusion.encoder import FusionVisionEncoder

        model = FusionVisionEncoder.load_from_checkpoint(checkpoint_path)
        return model, model.output_dim

    def _encode_image(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        tensor = self._image_tensor(batch)
        tensor = self._prepare_tensor(tensor)
        return self.model.encode_retrieval(tensor)

    def _image_tensor(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if isinstance(batch, dict):
            tensor = next(
                (
                    batch[key]
                    for key in self._TENSOR_KEYS[self.encoder]
                    if key in batch
                ),
                None,
            )
        else:
            tensor = batch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{self.encoder} vision encoder expects a tensor or a batch "
                f"dict containing one of {self._TENSOR_KEYS[self.encoder]}."
            )
        return tensor

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        if self.encoder == "geo" and tensor.max() > 1:
            return tensor / 127.5 - 1.0
        return tensor
