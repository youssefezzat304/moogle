"""encoder.py — native WAC vision encoder for LunarCLIP.

Wraps the pretrained WAC2WAC ViT encoder (trained on WAC image
reconstruction) and exposes the contract consumed by
lunar_clip.encoders.vision.lunar_vision_encoder.LunarVisionEncoder
(via its _load_wac / _encode_image path):

    output_dim: int
    encode_retrieval(batch) -> torch.Tensor   # shape (batch_size, output_dim)

The underlying transformer (MyTransformer) prepends a dedicated, learnable
retrieval token to the patch sequence — similar to a CLS token. This token
did not exist when the WAC2WAC checkpoint was trained, so it is randomly
initialised here. LunarVisionEncoder's freeze_encoder setting determines
whether the pretrained encoder weights remain frozen. The retrieval vector
returned by encode_retrieval() is this token's output, not a mean pool over
patch tokens.

MyTransformer.forward() only accepts its native image_size (256x256) since
it hardcodes that check for the reconstruction path. LunarCLIP feeds larger
patches (e.g. 512x512, see configs/clip/bpe_wac.yaml), so encode_retrieval()
here bypasses forward() and patchifies directly, bicubically resizing
pos_embed to the actual patch grid — mirroring
lunar_vision.model.geo.model.Encoder.encode_retrieval().
"""

from __future__ import annotations

from typing import Any

import torch

from lunar_vision.model.clip_backend import (
    VisionEncoderBackend, resize_pos_embed, source_pos_grid_size,
)
from lunar_vision.model.wac.config import (
    HIDDEN_DIM, IMAGE_SIZE, NUM_HEADS, NUM_LAYERS, VIT_PATCH_SIZE,
)
from lunar_vision.model.wac.transformer import MyTransformer


class WACEncoder(VisionEncoderBackend):
    """WAC ViT encoder whose freezing policy is owned by LunarVisionEncoder."""

    output_dim = HIDDEN_DIM

    def __init__(self) -> None:
        super().__init__()
        self.encoder = MyTransformer(
            patch_size=VIT_PATCH_SIZE,
            image_size=IMAGE_SIZE,
            img_channels=1,
            hidden_dim=HIDDEN_DIM,
            nheads=NUM_HEADS,
            num_layers=NUM_LAYERS,
        )

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "WACEncoder":
        """Load encoder weights from a WAC2WAC training checkpoint.

        Accepts either a full WAC2WAC checkpoint (dict with a "model" key
        containing both encoder.* and decoder.* weights) or an encoder-only
        state dict with no prefix.

        The checkpoint predates the retrieval token, so it has no
        "retrieval" key. That single missing key is expected and is left at
        its random initial value. Any other missing or unexpected key means
        a real mismatch, and raises instead of silently loading something
        wrong.
        """
        instance = cls()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
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

        instance.encoder.eval()
        return instance

    def encode_retrieval(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        """Return one retrieval vector per WAC patch.

        Deliberately NOT wrapped in torch.no_grad() so LunarVisionEncoder can
        train either the retrieval token alone or the full backend according
        to its freeze_encoder setting.

        Args:
            batch: a (B, 1, H, W) normalised WAC tensor, or a dict
                containing that tensor under the "tensor" key. H and W must
                be divisible by the encoder's patch size, but need not match
                the native training resolution (IMAGE_SIZE) — pos_embed is
                resized to fit.

        Returns:
            (B, output_dim) float tensor.
        """
        tensor = batch["tensor"] if isinstance(batch, dict) else batch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"WAC vision encoder expects a tensor, got {type(tensor)}.")
        if tensor.ndim != 4:
            raise ValueError(
                "WAC vision tensor must have shape (B, C, H, W), "
                f"got {tuple(tensor.shape)}."
            )

        patch_size = self.encoder.patch_size
        batch_size, _, height, width = tensor.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "WAC vision input height and width must be divisible by "
                f"the encoder patch size ({patch_size}). Got {(height, width)}."
            )

        tensor = tensor.float()
        x = self.encoder.conv_proj(tensor)
        _, hidden_dim, grid_h, grid_w = x.shape
        x = x.reshape(batch_size, hidden_dim, grid_h * grid_w).permute(0, 2, 1)

        pos_embed = self._resized_pos_embed((grid_h, grid_w))
        x = x + pos_embed.to(device=x.device, dtype=x.dtype)

        retrieval_tokens = self.encoder.retrieval.to(device=x.device, dtype=x.dtype).expand(
            batch_size, -1, -1,
        )
        x = torch.cat([retrieval_tokens, x], dim=1)

        tokens = self.encoder.encoder(x)  # MyTransformer's own TransformerEncoder
        return tokens[:, 0, :].float()

    def _resized_pos_embed(self, target_grid_size: tuple[int, int]) -> torch.Tensor:
        """Resize pos_embed to target_grid_size if it differs from the grid
        it was trained at (e.g. 16x16 -> 32x32 when moving from this
        encoder's native 256x256 training tile to CLIP's 512x512 patch with
        the same patch_size)."""
        source_grid_size = source_pos_grid_size(
            token_count=int(self.encoder.pos_embed.shape[1]),
            image_size=self.encoder.image_size,
            patch_size=self.encoder.patch_size,
        )
        return resize_pos_embed(
            self.encoder.pos_embed,
            source_grid_size=source_grid_size,
            target_grid_size=target_grid_size,
        )
