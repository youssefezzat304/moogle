from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from lunar_text.tokenizer.ngram.wrapper import NgramTokenizerWrapper
from lunar_text.utils.padding import pad_token_ids


REQUIRED_SPECIAL_TOKENS = ("[RETRIEVAL]", "[SOS]", "[EOS]", "[PAD]", "[MASK]", "[UNK]")


def load_tokenizer(tokenizer_path):

    tokenizer_path = Path(tokenizer_path)

    with open(tokenizer_path, "r") as f:
        config = json.load(f)

    model_type = config["model"]["type"]

    if model_type == "N-Gram":

        return NgramTokenizerWrapper(
            tokenizer_path
        )

    return Tokenizer.from_file(
        str(tokenizer_path)
    )


def special_token_ids(
    tokenizer_or_wrapper: Any,
    tokens: tuple[str, ...] = REQUIRED_SPECIAL_TOKENS,
    *,
    error_context: str = "Tokenizer",
) -> dict[str, int]:
    tokenizer = _as_tokenizer(tokenizer_or_wrapper)
    token_ids = {token: tokenizer.token_to_id(token) for token in tokens}
    missing = [token for token, token_id in token_ids.items() if token_id is None]
    if missing:
        raise ValueError(f"{error_context} is missing required special tokens: {missing}.")
    return {token: int(token_id) for token, token_id in token_ids.items()}


def special_token_id_list(
    tokenizer_or_wrapper: Any,
    tokens: tuple[str, ...] = REQUIRED_SPECIAL_TOKENS,
    *,
    require_all: bool = True,
    error_context: str = "Tokenizer",
) -> list[int]:
    tokenizer = _as_tokenizer(tokenizer_or_wrapper)
    ids: list[int] = []
    missing: list[str] = []
    for token in tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            missing.append(token)
        else:
            ids.append(int(token_id))

    if require_all and missing:
        raise ValueError(f"{error_context} is missing required special tokens: {missing}.")
    return ids


def validate_required_special_tokens(
    tokenizer_or_wrapper: Any,
    tokens: tuple[str, ...] = REQUIRED_SPECIAL_TOKENS,
) -> dict[str, int]:
    return special_token_ids(
        tokenizer_or_wrapper,
        tokens=tokens,
        error_context="Tokenizer",
    )


def _as_tokenizer(tokenizer_or_wrapper: Any) -> Tokenizer:
    return getattr(tokenizer_or_wrapper, "tokenizer", tokenizer_or_wrapper)
