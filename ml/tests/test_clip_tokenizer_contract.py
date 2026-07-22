from __future__ import annotations

import os
from pathlib import Path

import pytest
from tokenizers import Tokenizer


REQUIRED_SPECIAL_TOKENS = ("[RETRIEVAL]", "[SOS]", "[EOS]", "[PAD]", "[MASK]", "[UNK]")
DEFAULT_TOKENIZER_PATH = Path("artifacts/tokenizers/wordpiece/v3.0/tokenizer.json")


def _tokenizer_paths() -> list[Path]:
    configured_paths = os.environ.get("CLIP_TOKENIZER_PATHS") or os.environ.get(
        "CLIP_TOKENIZER_PATH"
    )
    if configured_paths:
        return [
            Path(path)
            for path in configured_paths.split(os.pathsep)
            if path.strip()
        ]
    if DEFAULT_TOKENIZER_PATH.exists():
        return [DEFAULT_TOKENIZER_PATH]
    return []


@pytest.mark.parametrize("tokenizer_path", _tokenizer_paths(), ids=str)
def test_tokenizer_is_compatible_with_clip_retrieval(tokenizer_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_ids = {
        token: tokenizer.token_to_id(token)
        for token in REQUIRED_SPECIAL_TOKENS
    }
    missing_tokens = [
        token
        for token, token_id in token_ids.items()
        if token_id is None
    ]

    assert not missing_tokens, (
        f"{tokenizer_path} is missing required CLIP special tokens: "
        f"{missing_tokens}"
    )

    encoded = tokenizer.encode(
        "cratered mare terrain boundary",
        add_special_tokens=False,
    )
    retrieval_ids = [
        token_ids["[RETRIEVAL]"],
        token_ids["[SOS]"],
        *encoded.ids,
        token_ids["[EOS]"],
    ]

    assert retrieval_ids[0] == token_ids["[RETRIEVAL]"]
    assert retrieval_ids[1] == token_ids["[SOS]"]
    assert retrieval_ids[-1] == token_ids["[EOS]"]
    assert token_ids["[RETRIEVAL]"] not in encoded.ids, (
        "The tokenizer should not inject [RETRIEVAL] when "
        "add_special_tokens=False; the CLIP backend owns that prefix."
    )


def test_at_least_one_tokenizer_path_was_selected() -> None:
    assert _tokenizer_paths(), (
        "No tokenizer path selected. Set CLIP_TOKENIZER_PATH to test one "
        "tokenizer or CLIP_TOKENIZER_PATHS to test multiple paths separated by "
        f"os.pathsep ({os.pathsep!r})."
    )
