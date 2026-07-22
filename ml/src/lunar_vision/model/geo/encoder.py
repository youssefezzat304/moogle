"""encoder.py — native Geo2Geo vision encoder for LunarCLIP.

Wraps the pretrained Geo2Geo ViT encoder (trained on geomap image
reconstruction) and exposes the contract consumed by
lunar_clip.encoders.vision.lunar_vision_encoder.LunarVisionEncoder
(via its _load_geo / _encode_image methods):

    output_dim: int
    encode_retrieval(batch) -> torch.Tensor   # shape (batch_size, output_dim)

The underlying Encoder (see model.py) has a dedicated, learnable retrieval
token prepended to the patch sequence — similar to a CLS token. This token
did not exist when the Geo2Geo checkpoint (best epoch: 50) was trained, so it
is randomly initialised here; it is left trainable while the rest of the
encoder is frozen by LunarVisionEncoder's freeze_encoder logic. The retrieval
vector returned by encode_retrieval() is this token's output, not a mean pool
over patch tokens.
"""

from __future__ import annotations

from typing import Any

import torch

from lunar_vision.model.clip_backend import VisionEncoderBackend
from lunar_vision.model.geo.model import Encoder


class GeoEncoder(VisionEncoderBackend):
    """Geo2Geo ViT encoder (no decoder) with a retrieval token."""

    def __init__(
        self,
        patch_size: int,
        image_size: int,
        hidden_dim: int,
        nheads: int,
        num_layers: int,
        img_channels: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            patch_size=patch_size,
            image_size=image_size,
            img_channels=img_channels,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_layers=num_layers,
        )
        self.output_dim = hidden_dim

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "GeoEncoder":
        """Load encoder weights from a Geo2Geo training checkpoint.

        The checkpoint is expected to be a dict with a "model" key (full
        Geo2Geo state dict, "encoder.*"/"decoder.*" prefixed) and an "args"
        key describing the architecture hyperparameters, as saved by
        lunar_vision.model.geo.train. Only "encoder."-prefixed keys are
        loaded; "decoder."-prefixed keys are ignored since CLIP never needs
        image reconstruction.

        The checkpoint predates the retrieval token, so it has no
        "retrieval" key. That single missing key is expected and is left at
        its random initial value. Any other missing or unexpected key means
        a real mismatch, and raises instead of silently loading something
        wrong.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})

        instance = cls(
            patch_size=int(args.get("enc_patch", 16)),
            image_size=int(args.get("patch_size", 256)),
            hidden_dim=int(args.get("hidden_dim", 512)),
            nheads=int(args.get("nheads", 8)),
            num_layers=int(args.get("num_layers", 6)),
        )

        state_dict = checkpoint["model"]
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            encoder_state = state_dict  # already encoder-only

        missing, unexpected = instance.encoder.load_state_dict(encoder_state, strict=False)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint, not present in model: {unexpected}")
        if [key for key in missing if key != "retrieval"]:
            raise RuntimeError(f"Missing keys other than the expected new 'retrieval' token: {missing}")

        return instance

    def encode_retrieval(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        """Return one retrieval vector per geomap patch.

        Deliberately NOT wrapped in torch.no_grad(). The retrieval token
        needs gradients to learn during CLIP training; every other
        parameter is frozen by LunarVisionEncoder's freeze_encoder logic, so
        frozen weights still won't update even though gradients pass through
        them on the way back to the retrieval token.

        Args:
            batch: a (B, 3, H, W) normalised geomap tensor, or a dict
                containing that tensor under one of the tensor keys handled
                by LunarVisionEncoder.

        Returns:
            (B, output_dim) float tensor.
        """
        tensor = batch["tensor"] if isinstance(batch, dict) else batch
        return self.encoder.encode_retrieval(tensor)
