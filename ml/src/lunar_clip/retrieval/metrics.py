from __future__ import annotations

import torch


RETRIEVAL_TOP_KS = (1, 5, 10)


def full_index_retrieval_metrics(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    text_patch_ids: torch.Tensor,
    image_patch_ids: torch.Tensor,
) -> dict[str, float]:
    if text_embeds.ndim != 2 or image_embeds.ndim != 2:
        raise ValueError("text_embeds and image_embeds must be 2D tensors.")
    if text_embeds.shape[1] != image_embeds.shape[1]:
        raise ValueError(
            "text_embeds and image_embeds must have the same embedding dimension. "
            f"Got {text_embeds.shape[1]} and {image_embeds.shape[1]}."
        )
    if text_embeds.shape[0] != text_patch_ids.numel():
        raise ValueError("text_patch_ids must contain one id per text embedding.")
    if image_embeds.shape[0] != image_patch_ids.numel():
        raise ValueError("image_patch_ids must contain one id per image embedding.")
    if text_embeds.shape[0] == 0 or image_embeds.shape[0] == 0:
        raise ValueError("Full-index retrieval metrics require non-empty embeddings.")

    scores = text_embeds.detach().cpu() @ image_embeds.detach().cpu().t()
    text_patch_ids = text_patch_ids.detach().cpu().reshape(-1)
    image_patch_ids = image_patch_ids.detach().cpu().reshape(-1)

    text_to_image_ranks = _first_correct_ranks(
        scores=scores,
        query_patch_ids=text_patch_ids,
        candidate_patch_ids=image_patch_ids,
    )
    image_to_text_ranks = _first_correct_ranks(
        scores=scores.t(),
        query_patch_ids=image_patch_ids,
        candidate_patch_ids=text_patch_ids,
    )

    metrics: dict[str, float] = {}
    recall_values: list[float] = []
    for top_k in RETRIEVAL_TOP_KS:
        text_to_image_recall = _recall_at_k(text_to_image_ranks, top_k)
        image_to_text_recall = _recall_at_k(image_to_text_ranks, top_k)
        metrics[f"full_text_to_image_recall@{top_k}"] = text_to_image_recall
        metrics[f"full_image_to_text_recall@{top_k}"] = image_to_text_recall
        recall_values.extend([text_to_image_recall, image_to_text_recall])

    metrics["full_text_to_image_median_rank"] = float(text_to_image_ranks.float().median().item())
    metrics["full_image_to_text_median_rank"] = float(image_to_text_ranks.float().median().item())
    metrics["full_text_to_image_mean_rank"] = float(text_to_image_ranks.float().mean().item())
    metrics["full_image_to_text_mean_rank"] = float(image_to_text_ranks.float().mean().item())
    metrics["full_mean_recall"] = float(sum(recall_values) / len(recall_values))
    return metrics


def text_to_image_retrieval_metrics(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    text_patch_ids: torch.Tensor,
    image_patch_ids: torch.Tensor,
) -> dict[str, float]:
    """Full-index text-to-image metrics for a selected set of text queries."""
    if text_embeds.ndim != 2 or image_embeds.ndim != 2:
        raise ValueError("text_embeds and image_embeds must be 2D tensors.")
    if text_embeds.shape[1] != image_embeds.shape[1]:
        raise ValueError("text_embeds and image_embeds must have the same embedding dimension.")
    if text_embeds.shape[0] != text_patch_ids.numel():
        raise ValueError("text_patch_ids must contain one id per text embedding.")
    if image_embeds.shape[0] != image_patch_ids.numel():
        raise ValueError("image_patch_ids must contain one id per image embedding.")
    if text_embeds.shape[0] == 0 or image_embeds.shape[0] == 0:
        raise ValueError("Full-index retrieval metrics require non-empty embeddings.")

    ranks = _first_correct_ranks(
        scores=text_embeds.detach().cpu() @ image_embeds.detach().cpu().t(),
        query_patch_ids=text_patch_ids.detach().cpu().reshape(-1),
        candidate_patch_ids=image_patch_ids.detach().cpu().reshape(-1),
    )
    metrics = {
        f"full_text_to_image_recall@{top_k}": _recall_at_k(ranks, top_k)
        for top_k in RETRIEVAL_TOP_KS
    }
    metrics["full_text_to_image_median_rank"] = float(ranks.float().median().item())
    metrics["full_text_to_image_mean_rank"] = float(ranks.float().mean().item())
    return metrics


def _first_correct_ranks(
    scores: torch.Tensor,
    query_patch_ids: torch.Tensor,
    candidate_patch_ids: torch.Tensor,
) -> torch.Tensor:
    sorted_indices = scores.argsort(dim=1, descending=True)
    sorted_candidate_ids = candidate_patch_ids[sorted_indices]
    matches = sorted_candidate_ids == query_patch_ids.unsqueeze(1)
    if not bool(matches.any(dim=1).all()):
        missing = query_patch_ids[~matches.any(dim=1)].tolist()
        raise ValueError(f"No matching candidate patch_id found for queries: {missing}")
    return matches.float().argmax(dim=1).long() + 1


def _recall_at_k(ranks: torch.Tensor, top_k: int) -> float:
    return float((ranks <= top_k).float().mean().item())
