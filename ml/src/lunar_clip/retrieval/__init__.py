import torch

from lunar_clip.retrieval.evaluation import (
    RetrievalEvaluationArtifact,
    RetrievalEvaluationMetadata,
    write_retrieval_evaluation_artifact,
)
from lunar_clip.retrieval.metrics import (
    full_index_retrieval_metrics,
    text_to_image_retrieval_metrics,
)
from lunar_clip.retrieval.vector_store import InMemoryVectorStore


def build_in_memory_index(
    ids: list[int | str],
    vectors: torch.Tensor,
) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(ids=ids, vectors=vectors)
    return store


def search_index(
    store: InMemoryVectorStore,
    query: torch.Tensor,
    top_k: int = 10,
) -> list[tuple[int | str, float]]:
    return store.search(query=query, top_k=top_k)

__all__ = [
    "InMemoryVectorStore",
    "RetrievalEvaluationArtifact",
    "RetrievalEvaluationMetadata",
    "build_in_memory_index",
    "full_index_retrieval_metrics",
    "text_to_image_retrieval_metrics",
    "search_index",
    "write_retrieval_evaluation_artifact",
]
