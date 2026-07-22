from __future__ import annotations

from typing import Any

import torch


def collate_clip_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    texts: list[str] = []
    text_patch_ids: list[int] = []
    caption_metadata: list[dict[str, str]] = []
    for sample in batch:
        sample_texts = sample["text"]
        sample_metadata = sample["caption_metadata"]
        if isinstance(sample_texts, str):
            sample_texts = [sample_texts]
        if isinstance(sample_metadata, dict):
            sample_metadata = [sample_metadata]
        if len(sample_texts) != len(sample_metadata):
            raise ValueError("Each caption must have one metadata record.")
        texts.extend(sample_texts)
        text_patch_ids.extend([sample["patch_id"]] * len(sample_texts))
        caption_metadata.extend(sample_metadata)

    image_patch_ids = torch.tensor(
        [sample["patch_id"] for sample in batch],
        dtype=torch.long,
    )
    output: dict[str, Any] = {
        "text": texts,
        "text_patch_id": torch.tensor(text_patch_ids, dtype=torch.long),
        "image_patch_id": image_patch_ids,
        # Preserve the original key for callers that identify image samples by
        # patch_id. New contrastive code uses the explicit modality-specific keys.
        "patch_id": image_patch_ids,
        "coords": torch.stack([sample["coords"] for sample in batch]),
        "caption_metadata": caption_metadata,
        "text_version": [metadata["source_version"] for metadata in caption_metadata],
    }
    if "vision" in batch[0]:
        modality = next(iter(batch[0]["vision"]))
        output["vision"] = {
            modality: {
                key: torch.stack([sample["vision"][modality][key] for sample in batch])
                for key in batch[0]["vision"][modality]
                if isinstance(batch[0]["vision"][modality][key], torch.Tensor)
            }
        }
    return output
