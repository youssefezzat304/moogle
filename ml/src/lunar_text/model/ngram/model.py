from __future__ import annotations

import torch
from torch import nn


class NgramMaskedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        mask_id: int,
        pad_id: int,
        max_length: int = 128,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.output_dim = d_model
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask = input_ids == self.pad_id
        _, seq_len = input_ids.shape
        if seq_len > self.max_length:
            raise ValueError(
                f"ngram input length {seq_len} exceeds max_length {self.max_length}."
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        logits = self.lm_head(hidden)
        return logits, hidden

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        _, hidden = self.forward(input_ids)
        return hidden
