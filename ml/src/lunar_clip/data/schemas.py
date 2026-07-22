from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CaptionPolicy = Literal[
    "first",
    "sample_one",
    "two_llm_descriptions",
]
SplitMode = Literal["train", "eval", "test"]

TWO_LLM_DESCRIPTION_SOURCES = (
    ("v1.0", "llm_description"),
    ("v2.0", "llm_description"),
)
MULTI_CAPTION_SOURCES = {
    "two_llm_descriptions": TWO_LLM_DESCRIPTION_SOURCES,
}

REQUIRED_COLUMNS = {
    "patch_id",
    "x_coord",
    "y_coord",
    "source_version",
    "prompt_style",
    "text",
}


@dataclass(frozen=True)
class CaptionRecord:
    text: str
    source_version: str
    prompt_style: str


@dataclass(frozen=True)
class PatchCaptions:
    patch_id: int
    x_coord: int
    y_coord: int
    captions: tuple[CaptionRecord, ...]
