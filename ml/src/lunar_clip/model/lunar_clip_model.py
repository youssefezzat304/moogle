from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lunar_clip.contracts.outputs import LunarCLIPOutput
from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.losses.contrastive import (
    multi_positive_contrastive_loss,
    symmetric_contrastive_loss,
)
from lunar_clip.model.logit_scale import LogitScale
from lunar_clip.utils import build_projection_head


class LunarCLIPModel(nn.Module):
    def __init__(
        self,
        text_adapter: LunarTextEncoder,
        vision_adapter: LunarVisionEncoder,
        projection_dim: int = 512,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.text_adapter = text_adapter
        self.vision_adapter = vision_adapter
        self.text_projection = build_projection_head(
            input_dim=text_adapter.output_dim,
            projection_dim=projection_dim,
        )
        self.image_projection = build_projection_head(
            input_dim=vision_adapter.output_dim,
            projection_dim=projection_dim,
        )
        self.logit_scale = LogitScale(temperature=temperature)

    def encode_text(self, text_batch) -> torch.Tensor:
        vectors = self.text_adapter.encode_text(text_batch).vectors
        return F.normalize(self.text_projection(vectors), dim=-1)

    def encode_image(self, image_batch, modality: str | None = None) -> torch.Tensor:
        modality = self._resolve_image_modality(modality)
        vectors = self.vision_adapter.encode_image(image_batch, modality=modality).vectors
        return F.normalize(self.image_projection(vectors), dim=-1)

    def forward(
        self,
        text_batch,
        image_batch,
        modality: str | None = None,
        return_loss: bool = True,
        text_patch_ids: torch.Tensor | None = None,
        image_patch_ids: torch.Tensor | None = None,
    ) -> LunarCLIPOutput:
        text_embeds = self.encode_text(text_batch)
        image_embeds = self.encode_image(image_batch, modality=modality)

        logit_scale = self.logit_scale()
        logits_per_text = logit_scale * text_embeds @ image_embeds.t()
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            if text_patch_ids is None and image_patch_ids is None:
                loss = symmetric_contrastive_loss(logits_per_text, logits_per_image)
            elif text_patch_ids is None or image_patch_ids is None:
                raise ValueError(
                    "Both text_patch_ids and image_patch_ids are required for "
                    "multi-positive contrastive loss."
                )
            else:
                loss = multi_positive_contrastive_loss(
                    logits_per_text=logits_per_text,
                    logits_per_image=logits_per_image,
                    text_patch_ids=text_patch_ids,
                    image_patch_ids=image_patch_ids,
                )

        return LunarCLIPOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            logit_scale=logit_scale,
        )

    def _resolve_image_modality(self, modality: str | None) -> str:
        if modality is not None:
            return modality
        expected_modality = getattr(self.vision_adapter, "expected_modality", None)
        if not isinstance(expected_modality, str):
            raise ValueError("Image modality must be provided for this vision adapter.")
        return expected_modality
