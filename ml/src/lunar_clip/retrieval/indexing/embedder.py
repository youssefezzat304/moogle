from __future__ import annotations

from typing import Any, Protocol

import torch

from lunar_clip.retrieval.indexing.contracts import EmbeddingDescriptor


class LunarClipImageModel(Protocol):
    def eval(self) -> Any: ...

    def to(self, device: torch.device) -> Any: ...

    def encode_image(
        self,
        image_batch: Any,
        modality: str | None = None,
    ) -> torch.Tensor: ...


class LunarClipImageEmbedder:
    """Indexing adapter for a fully reconstructed LunarCLIP model."""

    def __init__(
        self,
        *,
        model: LunarClipImageModel,
        descriptor: EmbeddingDescriptor,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self._descriptor = descriptor
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._descriptor

    def encode(self, images: Any) -> torch.Tensor:
        image_batch = _move_to_device(images, self.device)
        self.model.eval()
        with torch.inference_mode():
            embeddings = self.model.encode_image(
                image_batch,
                modality=self.descriptor.modality,
            )
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("LunarCLIP encode_image must return a tensor.")
        return embeddings


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value
