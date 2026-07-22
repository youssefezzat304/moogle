from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lunar_clip.contracts.batches import RetrievalBatch
from lunar_text.model.clip_backend import TextEncoderBackend


class LunarTextEncoder(nn.Module):

    def __init__(
        self,
        tokenizer_path: str,
        encoder: str = "bpe",
        checkpoint_path: str | None = None,
        model_config: Any | None = None,
        max_length: int | None = None,
        freeze_encoder: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder.lower()

        self.backend = _load_backend(
            encoder=self.encoder,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            max_length=max_length,
        )
        self.output_dim = self.backend.output_dim
        self.max_length = self.backend.max_length
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for parameter in self.backend.parameters():
                parameter.requires_grad = False
            self.backend.eval()

        self.device_override = torch.device(device) if device is not None else None
        if self.device_override is not None:
            self.to(self.device_override)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.backend.eval()
        return self

    def encode_text(self, texts: str | list[str]) -> RetrievalBatch:
        if not isinstance(texts, (str, list)):
            raise TypeError("LunarTextEncoder.encode_text expects a string or list of strings.")
        return RetrievalBatch(
            vectors=self.backend.encode_retrieval(texts),
            metadata={},
        )


def _load_backend(
    encoder: str,
    tokenizer_path: str,
    checkpoint_path: str | None,
    model_config: Any | None,
    max_length: int | None,
) -> TextEncoderBackend:
    if encoder == "bpe":
        from lunar_text.model.bpe.clip_adapter import BPECLIPTextBackend

        return BPECLIPTextBackend.load(
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            max_length=max_length,
        )
    if encoder == "wordpiece":
        from lunar_text.model.wordpiece.clip_adapter import WordPieceCLIPTextBackend

        return WordPieceCLIPTextBackend.load(
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            max_length=max_length,
        )
    if encoder == "ngram":
        from lunar_text.model.ngram.clip_adapter import NgramCLIPTextBackend

        return NgramCLIPTextBackend.load(
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            max_length=max_length,
        )
    raise ValueError(
        f"Unknown text encoder '{encoder}'. Available encoders: "
        "['bpe', 'ngram', 'wordpiece']"
    )
