from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LunarCLIPOutput:
    loss: torch.Tensor | None
    logits_per_text: torch.Tensor
    logits_per_image: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    logit_scale: torch.Tensor
