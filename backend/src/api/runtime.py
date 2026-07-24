from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from moogle_inference import RetrievalResult, load_retrieval_engine


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPOSITORY_ROOT / "storage/catalogs/lunar-v1"
INDEX_PATH = REPOSITORY_ROOT / "storage/indexes/bpe_geo/v1"
MODEL_MANIFEST_PATH = REPOSITORY_ROOT / "storage/models/bpe_geo/manifest.yaml"
MODEL_DEVICE = "cuda"


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
