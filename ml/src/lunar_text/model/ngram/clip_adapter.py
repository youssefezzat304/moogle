from __future__ import annotations

from typing import Any

import torch
from tokenizers import Tokenizer

from lunar_text.model.clip_backend import TextBackendBatch, TextEncoderBackend
from lunar_text.model.ngram.model import NgramMaskedTransformer
from lunar_text.utils.checkpoints import load_checkpoint
from lunar_text.utils.tokenizers import (
    REQUIRED_SPECIAL_TOKENS,
    load_tokenizer,
    pad_token_ids,
    special_token_ids,
)


class NgramCLIPTextBackend(TextEncoderBackend):
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: NgramMaskedTransformer,
        token_ids: dict[str, int],
        max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.token_ids = token_ids
        self.max_length = max_length
        self.pad_token_id = token_ids["[PAD]"]
        self.output_dim = int(model.output_dim)

    @classmethod
    def load(
        cls,
        tokenizer_path: str,
        checkpoint_path: str | None,
        max_length: int | None = None,
        model_config: Any | None = None,
    ) -> NgramCLIPTextBackend:
        del model_config
        if checkpoint_path is None:
            raise ValueError("NgramCLIPTextBackend requires checkpoint_path.")

        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "vocab" not in checkpoint:
            raise ValueError("ngram checkpoint must contain a 'vocab' dictionary.")
        tokenizer = load_tokenizer(tokenizer_path)
        token_ids = special_token_ids(tokenizer, REQUIRED_SPECIAL_TOKENS)

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("ngram checkpoint must contain 'model_state_dict'.")

        position_weight = state_dict.get("position_embedding.weight")
        token_weight = state_dict.get("token_embedding.weight")
        if not isinstance(position_weight, torch.Tensor) or not isinstance(token_weight, torch.Tensor):
            raise ValueError("ngram checkpoint is missing embedding weights.")

        resolved_max_length = max_length or int(position_weight.shape[0])
        output_dim = int(token_weight.shape[1])
        model = NgramMaskedTransformer(
            vocab_size=int(token_weight.shape[0]),
            mask_id=token_ids["[MASK]"],
            pad_id=token_ids["[PAD]"],
            max_length=resolved_max_length,
            d_model=output_dim,
        )
        model.load_state_dict(state_dict)

        if resolved_max_length < 3:
            raise ValueError("ngram max_length must be at least 3.")
        return cls(
            tokenizer=tokenizer,
            model=model,
            token_ids=token_ids,
            max_length=resolved_max_length,
        )

    def tokenize(self, texts: str | list[str]) -> TextBackendBatch:
        if isinstance(texts, str):
            texts = [texts]
        input_ids = torch.stack([self._encode_text(text) for text in texts])
        attention_mask = (input_ids != self.pad_token_id).long()
        return TextBackendBatch(input_ids=input_ids, attention_mask=attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.encode(input_ids=input_ids, attention_mask=attention_mask)

    @property
    def retrieval_token_id(self) -> int:
        return self.token_ids["[RETRIEVAL]"]

    def _encode_text(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [
            self.token_ids["[RETRIEVAL]"],
            self.token_ids["[SOS]"],
            *encoded.ids,
            self.token_ids["[EOS]"],
        ]
        if len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.token_ids["[EOS]"]]
        return pad_token_ids(
            ids,
            pad_token_id=self.pad_token_id,
            max_length=self.max_length,
        )
