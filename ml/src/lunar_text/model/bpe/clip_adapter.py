from __future__ import annotations

from typing import Any

import torch
from tokenizers import Tokenizer

from lunar_text.model.bpe.config import ModelConfig
from lunar_text.model.bpe.model import BPELunarMLM
from lunar_text.model.clip_backend import TextBackendBatch, TextEncoderBackend
from lunar_text.utils.checkpoints import (
    checkpoint_value,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)
from lunar_text.utils.tokenizers import (
    REQUIRED_SPECIAL_TOKENS,
    load_tokenizer,
    pad_token_ids,
    special_token_ids,
)


class BPECLIPTextBackend(TextEncoderBackend):
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: BPELunarMLM,
        token_ids: dict[str, int],
        max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.token_ids = token_ids
        self.max_length = max_length
        self.pad_token_id = token_ids["[PAD]"]
        self.output_dim = int(model.config.embed_dim)

    @classmethod
    def load(
        cls,
        tokenizer_path: str,
        checkpoint_path: str | None,
        model_config: Any | None = None,
        max_length: int | None = None,
    ) -> BPECLIPTextBackend:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        config = _resolve_model_config(checkpoint=checkpoint, model_config=model_config)
        tokenizer = load_tokenizer(tokenizer_path)
        token_ids = special_token_ids(tokenizer, REQUIRED_SPECIAL_TOKENS)

        if checkpoint is None:
            tokenizer_vocab_size = int(tokenizer.get_vocab_size())
            config.vocab_size = tokenizer_vocab_size
            config.pad_token_id = token_ids["[PAD]"]

        model = BPELunarMLM(config)
        if checkpoint is not None:
            model.load_state_dict(model_state_dict_from_checkpoint(checkpoint))

        resolved_max_length = max_length or int(config.max_seq_len)
        if resolved_max_length < 3:
            raise ValueError("BPE max_length must be at least 3.")
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


def _resolve_model_config(
    checkpoint: dict[str, Any] | None,
    model_config: Any | None,
) -> ModelConfig:
    config_payload = checkpoint_value(checkpoint, "model_config")
    config = model_config or (ModelConfig(**config_payload) if config_payload else None)
    if config is None:
        raise ValueError(
            "BPECLIPTextBackend requires either checkpoint_path with "
            "model_config metadata or an explicit model_config."
        )
    return config
