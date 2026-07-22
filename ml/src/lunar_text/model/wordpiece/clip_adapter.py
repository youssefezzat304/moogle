from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from torch import nn

from lunar_text.model.clip_backend import TextBackendBatch, TextEncoderBackend
from lunar_text.utils.checkpoints import checkpoint_value, load_checkpoint
from lunar_text.utils.tokenizers import (
    REQUIRED_SPECIAL_TOKENS,
    load_tokenizer,
    pad_token_ids,
    special_token_ids,
)


class WordPieceCLIPTextBackend(TextEncoderBackend):
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: _BertTextModel,
        token_ids: dict[str, int],
        max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.token_ids = token_ids
        self.max_length = max_length
        self.pad_token_id = token_ids["[PAD]"]
        self.output_dim = int(model.config.hidden_size)

    @classmethod
    def load(
        cls,
        tokenizer_path: str,
        checkpoint_path: str | None,
        model_config: Any | None = None,
        max_length: int | None = None,
    ) -> WordPieceCLIPTextBackend:
        if checkpoint_path is None:
            raise ValueError("WordPieceCLIPTextBackend requires checkpoint_path.")

        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        tokenizer = load_tokenizer(tokenizer_path)
        token_ids = special_token_ids(tokenizer, REQUIRED_SPECIAL_TOKENS)
        config_payload = _resolve_config_payload(
            checkpoint=checkpoint,
            model_config=model_config,
            checkpoint_path=checkpoint_path,
        )
        bert_config = _build_bert_config(
            config_payload,
            tokenizer.get_vocab_size(),
            checkpoint_type_vocab_size=_checkpoint_type_vocab_size(checkpoint),
        )
        model = _BertTextModel(bert_config)
        _load_weights(model, checkpoint)

        resolved_max_length = max_length or int(bert_config.max_position_embeddings)
        if resolved_max_length < 3:
            raise ValueError("WordPiece max_length must be at least 3.")
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


class _BertTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        from transformers import BertModel

        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


def _resolve_config_payload(
    checkpoint: dict[str, Any] | None,
    model_config: Any | None,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    if model_config is not None:
        return _config_payload(model_config)
    for key in ("model_config", "config", "hyper_parameters"):
        payload = checkpoint_value(checkpoint, key)
        if payload:
            return _config_payload(payload)
    if checkpoint_path is not None:
        config_path = Path(checkpoint_path).parent / "config.json"
        if config_path.exists():
            import json

            return _config_payload(json.loads(config_path.read_text(encoding="utf-8")))
    raise ValueError(
        "WordPieceCLIPTextBackend requires model_config metadata in the checkpoint, "
        "a sibling config.json, or an explicit model_config."
    )


def _config_payload(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if hasattr(config, "__dict__") and not isinstance(config, dict):
        return dict(config.__dict__)
    if not isinstance(config, dict):
        raise TypeError("model_config must be a dict-like object.")
    return dict(config)


def _build_bert_config(
    payload: dict[str, Any],
    tokenizer_vocab_size: int,
    checkpoint_type_vocab_size: int | None = None,
):
    from transformers import BertConfig

    model_payload = dict(payload.get("model") or payload)
    training_payload = dict(payload.get("training") or {})
    max_positions = int(
        model_payload.get(
            "max_position_embeddings",
            training_payload.get("max_seq_len", 128),
        )
    )
    type_vocab_size = int(
        model_payload.get(
            "type_vocab_size",
            training_payload.get(
                "type_vocab_size",
                checkpoint_type_vocab_size or 1,
            ),
        )
    )
    return BertConfig(
        vocab_size=int(model_payload.get("vocab_size", tokenizer_vocab_size)),
        hidden_size=int(model_payload.get("hidden_size", 256)),
        num_hidden_layers=int(model_payload.get("num_hidden_layers", 4)),
        num_attention_heads=int(model_payload.get("num_attention_heads", 4)),
        intermediate_size=int(model_payload.get("intermediate_size", 512)),
        max_position_embeddings=max_positions,
        type_vocab_size=type_vocab_size,
        pad_token_id=int(model_payload.get("pad_token_id", 0)),
    )


def _checkpoint_state_dict(
    checkpoint: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    if checkpoint is None:
        raise ValueError("WordPieceCLIPTextBackend requires checkpoint weights.")

    raw_state = (
        checkpoint.get("model")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
    )
    if not isinstance(raw_state, dict):
        raise ValueError("WordPiece checkpoint does not contain model weights.")
    return raw_state


def _checkpoint_type_vocab_size(
    checkpoint: dict[str, Any] | None,
) -> int | None:
    if checkpoint is None:
        return None
    for key, value in _checkpoint_state_dict(checkpoint).items():
        if (
            key.endswith("embeddings.token_type_embeddings.weight")
            and isinstance(value, torch.Tensor)
            and value.ndim == 2
        ):
            return int(value.shape[0])
    return None


def _load_weights(model: _BertTextModel, checkpoint: dict[str, Any] | None) -> None:
    raw_state = _checkpoint_state_dict(checkpoint)

    state = {}
    for key, value in raw_state.items():
        clean_key = key
        for prefix in ("model.", "bert."):
            clean_key = clean_key.removeprefix(prefix)
        state[f"bert.{clean_key}"] = value
    model.load_state_dict(state, strict=False)
