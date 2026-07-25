from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from moogle_inference import RetrievalResult, load_retrieval_engine


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse_allowed_origins(value: str) -> tuple[str, ...]:
    origins = tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "MOOGLE_ALLOWED_ORIGINS must contain comma-separated HTTP(S) origins."
            )
    return origins


def _artifact_path(variable: str, default: str) -> Path:
    configured = os.environ.get(variable, default)
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


CATALOG_PATH = _artifact_path(
    "MOOGLE_CATALOG_PATH",
    "storage/catalogs/lunar-v1",
)
INDEX_PATH = _artifact_path(
    "MOOGLE_INDEX_PATH",
    "storage/indexes/bpe_geo/v1",
)
MODEL_MANIFEST_PATH = _artifact_path(
    "MOOGLE_MODEL_MANIFEST_PATH",
    "storage/models/bpe_geo/manifest.yaml",
)
MODEL_DEVICE = os.environ.get("MOOGLE_MODEL_DEVICE", "cpu").strip() or "cpu"
ALLOWED_ORIGINS = _parse_allowed_origins(os.environ.get("MOOGLE_ALLOWED_ORIGINS", ""))


class RetrievalService(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def index_size(self) -> int: ...

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]: ...

    def wac_image_path(self, patch_id: int) -> Path: ...


EngineLoader = Callable[[], RetrievalService]


def load_production_service() -> RetrievalService:
    return load_retrieval_engine(
        catalog_path=CATALOG_PATH,
        index_path=INDEX_PATH,
        model_manifest_path=MODEL_MANIFEST_PATH,
        device=MODEL_DEVICE,
    )
