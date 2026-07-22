from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(
    logits_per_text: torch.Tensor,
    logits_per_image: torch.Tensor,
) -> torch.Tensor:
    if logits_per_text.shape != logits_per_image.t().shape:
        raise ValueError(
            "Expected logits_per_image to be the transpose shape of logits_per_text. "
            f"Got {tuple(logits_per_text.shape)} and {tuple(logits_per_image.shape)}."
        )
        
    labels = torch.arange(logits_per_text.shape[0], device=logits_per_text.device)
    
    text_loss = F.cross_entropy(logits_per_text, labels)
    image_loss = F.cross_entropy(logits_per_image, labels)
    
    return (text_loss + image_loss) / 2.0


def multi_positive_contrastive_loss(
    logits_per_text: torch.Tensor,
    logits_per_image: torch.Tensor,
    text_patch_ids: torch.Tensor,
    image_patch_ids: torch.Tensor,
) -> torch.Tensor:
    """Symmetric CLIP loss with positives defined by matching patch IDs."""
    if logits_per_text.ndim != 2 or logits_per_image.ndim != 2:
        raise ValueError("Contrastive logits must be 2D tensors.")
    if logits_per_text.shape != logits_per_image.t().shape:
        raise ValueError(
            "Expected logits_per_image to be the transpose shape of logits_per_text. "
            f"Got {tuple(logits_per_text.shape)} and {tuple(logits_per_image.shape)}."
        )

    text_patch_ids = text_patch_ids.to(logits_per_text.device).reshape(-1)
    image_patch_ids = image_patch_ids.to(logits_per_text.device).reshape(-1)
    if text_patch_ids.numel() != logits_per_text.shape[0]:
        raise ValueError("text_patch_ids must contain one ID per text logit row.")
    if image_patch_ids.numel() != logits_per_text.shape[1]:
        raise ValueError("image_patch_ids must contain one ID per image logit column.")

    text_positive_mask = text_patch_ids[:, None] == image_patch_ids[None, :]
    image_positive_mask = text_positive_mask.t()
    _validate_positive_mask(text_positive_mask, anchor_name="text")
    _validate_positive_mask(image_positive_mask, anchor_name="image")

    text_loss = _multi_positive_cross_entropy(
        logits=logits_per_text,
        positive_mask=text_positive_mask,
    )
    image_loss = _multi_positive_cross_entropy(
        logits=logits_per_image,
        positive_mask=image_positive_mask,
    )
    return (text_loss + image_loss) / 2.0


def _multi_positive_cross_entropy(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    targets = positive_mask.to(dtype=logits.dtype)
    targets = targets / targets.sum(dim=1, keepdim=True)
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _validate_positive_mask(
    positive_mask: torch.Tensor,
    anchor_name: str,
) -> None:
    missing = ~positive_mask.any(dim=1)
    if bool(missing.any()):
        missing_indices = missing.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Every {anchor_name} anchor requires at least one positive; "
            f"missing positives for indices {missing_indices}."
        )
