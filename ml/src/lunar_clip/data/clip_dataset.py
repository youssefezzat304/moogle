from __future__ import annotations

from typing import Any

import random

import polars as pl
import torch
from torch.utils.data import Dataset

from lunar_clip.data.collate import collate_clip_batch
from lunar_clip.data.schemas import (
    CaptionPolicy,
    CaptionRecord,
    MULTI_CAPTION_SOURCES,
    PatchCaptions,
    REQUIRED_COLUMNS,
    SplitMode,
)


def select_captions(
    patch: PatchCaptions,
    index: int,
    dataset_size: int,
    caption_policy: CaptionPolicy,
    split: SplitMode,
    seed: int,
    epoch: int,
) -> tuple[CaptionRecord, ...]:
    if caption_policy in MULTI_CAPTION_SOURCES:
        return _select_required_captions(patch, caption_policy)
    if caption_policy == "first" or split != "train":
        return (patch.captions[0],)
    rng = random.Random(seed + epoch * dataset_size + index)
    return (rng.choice(patch.captions),)


def select_caption(
    patch: PatchCaptions,
    index: int,
    dataset_size: int,
    caption_policy: CaptionPolicy,
    split: SplitMode,
    seed: int,
    epoch: int,
) -> CaptionRecord:
    """Backward-compatible single-caption selector for legacy policies."""
    captions = select_captions(
        patch=patch,
        index=index,
        dataset_size=dataset_size,
        caption_policy=caption_policy,
        split=split,
        seed=seed,
        epoch=epoch,
    )
    if len(captions) != 1:
        raise ValueError(
            f"Caption policy '{caption_policy}' selects {len(captions)} captions; "
            "use select_captions instead."
        )
    return captions[0]


class LunarCLIPDataset(Dataset):
    def __init__(
        self,
        captions_path: str,
        vision_dataset: Dataset | None = None,
        modality: str = "geomap",
        caption_policy: CaptionPolicy = "sample_one",
        split: SplitMode = "train",
        seed: int = 42,
        eval_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if caption_policy not in ("sample_one", "first", *MULTI_CAPTION_SOURCES):
            raise ValueError(f"Unsupported caption_policy: {caption_policy}")
        if split not in ("train", "eval", "test"):
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 <= eval_fraction < 1.0:
            raise ValueError(
                "eval_fraction must be greater than or equal to 0 and less than 1."
            )

        self.patches = _select_split_patches(
            patches=_load_patch_captions(captions_path),
            split=split,
            eval_fraction=eval_fraction,
            seed=seed,
        )
        self.vision_dataset = vision_dataset
        self.modality = modality
        self.caption_policy = caption_policy
        self.split = split
        self.seed = seed
        self.epoch = 0
        if caption_policy in MULTI_CAPTION_SOURCES:
            for patch in self.patches:
                _select_required_captions(patch, caption_policy)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patch = self.patches[index]
        captions = select_captions(
            patch=patch,
            index=index,
            dataset_size=len(self.patches),
            caption_policy=self.caption_policy,
            split=self.split,
            seed=self.seed,
            epoch=self.epoch,
        )
        caption_metadata = [
            {
                "source_version": caption.source_version,
                "prompt_style": caption.prompt_style,
            }
            for caption in captions
        ]
        sample: dict[str, Any] = {
            "text": (
                captions[0].text
                if len(captions) == 1
                else [caption.text for caption in captions]
            ),
            "patch_id": patch.patch_id,
            "coords": torch.tensor([patch.x_coord, patch.y_coord], dtype=torch.long),
            "caption_metadata": (
                caption_metadata[0] if len(caption_metadata) == 1 else caption_metadata
            ),
        }

        if self.vision_dataset is not None:
            image_sample = self.vision_dataset[int(patch.patch_id)]
            if self.modality not in image_sample:
                raise KeyError(
                    f"Vision dataset sample for patch {patch.patch_id} does not "
                    f"contain modality '{self.modality}'."
                )
            sample["vision"] = {self.modality: image_sample[self.modality]}

        return sample

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_clip_batch(batch)


def _select_required_captions(
    patch: PatchCaptions,
    caption_policy: CaptionPolicy,
) -> tuple[CaptionRecord, ...]:
    selected: list[CaptionRecord] = []
    for source_version, prompt_style in MULTI_CAPTION_SOURCES[caption_policy]:
        matches = [
            caption
            for caption in patch.captions
            if caption.source_version == source_version
            and caption.prompt_style == prompt_style
            ]
        if len(matches) != 1:
            raise ValueError(
                f"Caption policy '{caption_policy}' requires exactly one "
                f"{source_version}/{prompt_style} caption for patch "
                f"{patch.patch_id}; found {len(matches)}."
            )
        selected.append(matches[0])
    return tuple(selected)


def _load_patch_captions(captions_path: str) -> list[PatchCaptions]:
    df = pl.read_parquet(captions_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Caption parquet is missing required columns: {sorted(missing)}. "
            f"Found columns: {df.columns}"
        )

    cleaned = (
        df.with_columns(pl.col("text").cast(pl.Utf8).str.strip_chars().alias("text"))
        .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
        .sort(["patch_id", "source_version", "prompt_style"])
    )
    patches: list[PatchCaptions] = []
    for patch_df in cleaned.partition_by("patch_id", maintain_order=True):
        rows = patch_df.to_dicts()
        first = rows[0]
        captions = tuple(
            CaptionRecord(
                text=str(row["text"]),
                source_version=str(row["source_version"]),
                prompt_style=str(row["prompt_style"]),
            )
            for row in rows
        )
        patches.append(
            PatchCaptions(
                patch_id=int(first["patch_id"]),
                x_coord=int(first["x_coord"]),
                y_coord=int(first["y_coord"]),
                captions=captions,
            )
        )
    return patches


def _select_split_patches(
    patches: list[PatchCaptions],
    split: SplitMode,
    eval_fraction: float,
    seed: int,
) -> list[PatchCaptions]:
    if split == "test" or eval_fraction == 0.0:
        return patches
    if len(patches) < 2:
        return [] if split == "eval" else patches

    indices = list(range(len(patches)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, round(len(indices) * eval_fraction))
    eval_size = min(eval_size, len(indices) - 1)
    eval_indices = set(indices[:eval_size])

    if split == "eval":
        return [patch for index, patch in enumerate(patches) if index in eval_indices]
    return [patch for index, patch in enumerate(patches) if index not in eval_indices]
