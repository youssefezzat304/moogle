from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    checkpoint_path: str | Path | None,
    map_location: str | torch.device,
) -> dict[str, Any] | None:
    if checkpoint_path is None or str(checkpoint_path) == "":
        return None

    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_value(checkpoint: dict[str, Any] | None, key: str) -> Any:
    if checkpoint is None:
        return None
    if key in checkpoint:
        return checkpoint[key]
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    return hyper_parameters.get(key)


def model_state_dict_from_checkpoint(
    checkpoint: dict[str, Any],
    lightning_prefix: str = "model.",
) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    state_dict = checkpoint.get("state_dict")
    model_state = {
        key.removeprefix(lightning_prefix): value
        for key, value in state_dict.items()
        if key.startswith(lightning_prefix)
    }
    return model_state
