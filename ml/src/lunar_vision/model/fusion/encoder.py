"""encoder.py — WAC-side retrieval encoder from the joint fusion model, for LunarCLIP.

Wraps the WAC-side half (`enc_w`) of a trained `JointFusion` checkpoint and
exposes the contract consumed by
lunar_clip.encoders.vision.lunar_vision_encoder.LunarVisionEncoder (via its
_load_fusion / _encode_image path):

    output_dim: int
    encode_retrieval(batch) -> torch.Tensor   # shape (batch_size, output_dim)

`JointFusion` was trained from scratch on the wac2geo/geo2wac/wac2wac/geo2geo
paths and has no retrieval token — it produces 256 patch tokens per image, not
one pooled retrieval vector. Only `enc_w` (the WAC encoder) is used here; the
geo encoder and both decoders are training/reconstruction machinery LunarCLIP
does not need, same as how the Geo2Geo adapter drops its decoder.

The retrieval token is new — it did not exist when the joint fusion checkpoint
was trained, so it is randomly initialised here. Whether every other `enc_w`
weight stays frozen at its pretrained value while the retrieval token trains
is governed entirely by LunarVisionEncoder's freeze_encoder config flag (see
lunar_clip.encoders.vision.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lunar_vision.model.clip_backend import VisionEncoderBackend
from lunar_vision.model.fusion.model import ViTEncoder


class FusionVisionEncoder(VisionEncoderBackend):
    """WAC-side joint-fusion encoder with a retrieval token. Whether every
    other weight stays frozen at its pretrained joint-fusion value is decided
    by LunarVisionEncoder's freeze_encoder config flag, not by this class."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.encoder = ViTEncoder(
            in_channels=1,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_layers=num_layers,
        )

        # The retrieval token is new; it did not exist when the joint-fusion
        # checkpoint was trained, so it is randomly initialised here and must
        # be learned separately (see lunar_vision.model.fusion.encoder /
        # LunarVisionEncoder's freeze_encoder handling for how training
        # decides which parameters, including this one, stay trainable).
        self.encoder.retrieval = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "FusionVisionEncoder":
        """Load enc_w weights from a JointFusion training checkpoint.

        Accepts a full JointFusion checkpoint (dict with a "model" key
        containing enc_w.*, enc_g.*, dec_w.*, dec_g.* weights). Only the
        "enc_w." keys are used; every other prefix is ignored since only the
        WAC-side encoder is needed for CLIP retrieval.

        The checkpoint predates the retrieval token, so it has no
        "enc_w.retrieval" key. That single missing key is expected and is
        left at its random initial value. Any other missing or unexpected key
        means a real mismatch, and raises instead of silently loading
        something wrong.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        instance = cls(
            image_size=int(args.get("image_size", 512)),
            patch_size=int(args.get("patch_size", 32)),
            hidden_dim=int(args.get("hidden_dim", 512)),
            nheads=int(args.get("nheads", 8)),
            num_layers=int(args.get("num_layers", 6)),
        )

        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        encoder_state = {
            key[len("enc_w."):]: value
            for key, value in state_dict.items()
            if key.startswith("enc_w.")
        }
        if not encoder_state:
            raise RuntimeError(
                "No 'enc_w.' keys found in checkpoint; expected a JointFusion "
                "state dict with an enc_w submodule."
            )

        missing, unexpected = instance.encoder.load_state_dict(encoder_state, strict=False)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint, not present in model: {unexpected}")
        if [key for key in missing if key != "retrieval"]:
            raise RuntimeError(f"Missing keys other than the expected new 'retrieval' token: {missing}")

        instance.encoder.eval()
        return instance

    def encode_retrieval(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        """Return one retrieval vector per WAC patch.

        Deliberately NOT wrapped in torch.no_grad(). The retrieval token
        needs gradients to learn during CLIP training; every other
        parameter already has requires_grad=False, so frozen weights still
        won't update even though gradients pass through them on the way
        back to the retrieval token.

        All ViT-specific mechanics (patchification, positional-embedding
        handling, retrieval-token preparation, transformer execution, and
        retrieval-token pooling) live here rather than in the CLIP layer,
        so LunarCLIP never has to reach into enc_w's internals.

        Unlike Geo, Fusion has one fixed native resolution (image_size x
        image_size, matching how the joint-fusion checkpoint was trained)
        and does not interpolate its positional embeddings to other
        resolutions — a mismatched input raises a ValueError instead.

        Args:
            batch: a (B, 1, image_size, image_size) normalised WAC tensor,
                or a dict containing that tensor under the "tensor" key.

        Returns:
            (B, output_dim) float tensor.
        """
        tensor = batch["tensor"] if isinstance(batch, dict) else batch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Fusion vision encoder expects a tensor, got {type(tensor)}.")
        if tensor.ndim != 4:
            raise ValueError(
                "Fusion vision tensor must have shape (B, C, H, W), "
                f"got {tuple(tensor.shape)}."
            )

        patch_size = self.encoder.patch_size
        _, _, height, width = tensor.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "Fusion vision input height and width must be divisible by "
                f"the encoder patch size ({patch_size}). Got {(height, width)}."
            )

        tensor = tensor.float()
        x = self.encoder.conv_proj(tensor)
        batch_size, hidden_dim, grid_h, grid_w = x.shape

        num_patches = int(self.encoder.pos_embed.shape[1])
        if grid_h * grid_w != num_patches:
            expected_side = int(self.encoder.image_size) // patch_size
            raise ValueError(
                "Fusion vision encoder was trained for a fixed "
                f"{expected_side}x{expected_side} patch grid "
                f"({self.encoder.image_size}x{self.encoder.image_size} input, "
                f"patch size {patch_size}), but got a {grid_h}x{grid_w} grid "
                f"from input shape {(height, width)}. Fusion does not support "
                "resizing its positional embeddings to other resolutions; "
                f"use a {self.encoder.image_size}x{self.encoder.image_size} input instead."
            )

        x = x.reshape(batch_size, hidden_dim, grid_h * grid_w).permute(0, 2, 1)
        x = x + self.encoder.pos_embed.to(device=x.device, dtype=x.dtype)

        retrieval_tokens = self.encoder.retrieval.to(device=x.device, dtype=x.dtype).expand(
            batch_size, -1, -1,
        )
        x = torch.cat([retrieval_tokens, x], dim=1)

        tokens = self.encoder.transformer(x)
        return tokens[:, 0, :].float()
