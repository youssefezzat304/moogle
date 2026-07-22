from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TextBackendBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class TextEncoderBackend(nn.Module):
    output_dim: int
    max_length: int
    pad_token_id: int

    def tokenize(self, texts: str | list[str]) -> TextBackendBatch:
        raise NotImplementedError

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def encode_retrieval(self, texts: str | list[str]) -> torch.Tensor:
        """Encode text into one retrieval vector per input.

        Backends may override this method when their retrieval preparation or
        pooling differs. The default preserves the existing tokenizer, encoder,
        and retrieval-token pooling contract while providing a single public
        entry point for LunarCLIP.
        """
        token_batch = self.tokenize(texts)
        device = self._device
        input_ids = token_batch.input_ids.to(device)
        attention_mask = token_batch.attention_mask.to(device)
        hidden_states = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        vectors = self.pool(
            hidden_states=hidden_states,
            input_ids=input_ids,
        )
        if vectors.ndim != 2:
            raise ValueError(
                "encode_retrieval must return a 2D tensor shaped "
                "(batch_size, hidden_dim)."
            )
        return vectors.float()

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        expected_token = self.retrieval_token_id
        if not torch.all(input_ids[:, 0] == expected_token):
            raise ValueError("text retrieval requires [RETRIEVAL] at index 0.")
        return hidden_states[:, 0, :]

    @property
    def retrieval_token_id(self) -> int:
        raise NotImplementedError

    @property
    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
