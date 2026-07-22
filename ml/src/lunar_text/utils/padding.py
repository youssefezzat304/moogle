import torch


def pad_token_ids(
    ids: list[int],
    pad_token_id: int,
    max_length: int,
) -> torch.Tensor:

    ids = ids + [pad_token_id] * (max_length - len(ids))

    return torch.tensor(
        ids,
        dtype=torch.long,
    )