from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder


TEXT_ENCODER_ARTIFACTS = (
    (
        "bpe",
        Path("artifacts/tokenizers/bpe/v4.0/tokenizer.json"),
        Path("artifacts/text_models/bpe/step_085000.ckpt"),
    ),
    (
        "wordpiece",
        Path("artifacts/tokenizers/wordpiece/v3.0/tokenizer.json"),
        Path("artifacts/text_models/wordpiece/wordpiece_mlm-epoch=98-train_loss=0.454.ckpt"),
    ),
    (
        "ngram",
        Path("artifacts/tokenizers/ngram/v1.0/tokenizer.json"),
        Path("artifacts/text_models/ngram/checkpoint_best.pth"),
    ),
)


@pytest.mark.parametrize(
    ("encoder_name", "tokenizer_path", "checkpoint_path"),
    TEXT_ENCODER_ARTIFACTS,
    ids=[case[0] for case in TEXT_ENCODER_ARTIFACTS],
)
def test_text_encoder_checkpoint_loads_and_backpropagates(
    encoder_name: str,
    tokenizer_path: Path,
    checkpoint_path: Path,
) -> None:
    if not tokenizer_path.is_file() or not _materialized_checkpoint(checkpoint_path):
        pytest.skip(f"Artifacts are not materialized for {encoder_name}.")

    encoder = LunarTextEncoder(
        encoder=encoder_name,
        tokenizer_path=str(tokenizer_path),
        checkpoint_path=str(checkpoint_path),
        freeze_encoder=False,
    )
    vectors = encoder.encode_text(
        ["A cratered lunar plain.", "A smooth highland patch."]
    ).vectors

    assert vectors.shape == (2, encoder.output_dim)
    if encoder_name == "wordpiece":
        assert encoder.backend.model.config.type_vocab_size == 2
    vectors.square().mean().backward()

    gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def _materialized_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as checkpoint:
        return not checkpoint.read(64).startswith(b"version https://git-lfs.github.com")
