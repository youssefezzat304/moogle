from __future__ import annotations
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM

from lunar_text.model.bpe.config import ModelConfig


class BPELunarMLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BertForMaskedLM(_bert_config_from_model_config(config))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        if labels is not None:
            if not (labels != -100).any():
                logits = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                return logits.sum() * 0.0, logits

            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            return outputs.loss, outputs.logits

        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        return self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


def _bert_config_from_model_config(config: ModelConfig) -> BertConfig:
    return BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.embed_dim,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        intermediate_size=config.ffn_dim,
        hidden_act="gelu",
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        max_position_embeddings=config.max_seq_len,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=config.layer_norm_eps,
        pad_token_id=config.pad_token_id,
        tie_word_embeddings=True,
    )
