from __future__ import annotations

import torch

from lunar_clip.contracts.outputs import LunarCLIPOutput


def in_batch_top1_retrieval_metrics(
    output: LunarCLIPOutput,
    text_patch_ids: torch.Tensor | None = None,
    image_patch_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = output.logits_per_text.device
    if text_patch_ids is None and image_patch_ids is None:
        if output.logits_per_text.shape[0] != output.logits_per_image.shape[0]:
            raise ValueError(
                "Patch IDs are required when text and image batch sizes differ."
            )
        text_patch_ids = torch.arange(output.logits_per_text.shape[0], device=device)
        image_patch_ids = torch.arange(output.logits_per_image.shape[0], device=device)
    elif text_patch_ids is None or image_patch_ids is None:
        raise ValueError("Both text_patch_ids and image_patch_ids are required.")
    text_patch_ids = text_patch_ids.to(device).reshape(-1)
    image_patch_ids = image_patch_ids.to(device).reshape(-1)
    if text_patch_ids.numel() != output.logits_per_text.shape[0]:
        raise ValueError("text_patch_ids must contain one ID per text embedding.")
    if image_patch_ids.numel() != output.logits_per_image.shape[0]:
        raise ValueError("image_patch_ids must contain one ID per image embedding.")

    predicted_image_ids = image_patch_ids[output.logits_per_text.argmax(dim=1)]
    predicted_text_ids = text_patch_ids[output.logits_per_image.argmax(dim=1)]
    text_to_image = (predicted_image_ids == text_patch_ids).float().mean()
    image_to_text = (predicted_text_ids == image_patch_ids).float().mean()
    return {
        "in_batch_text_to_image_top1": text_to_image,
        "in_batch_image_to_text_top1": image_to_text,
        "in_batch_retrieval_top1": (text_to_image + image_to_text) / 2.0,
    }
