from __future__ import annotations
import torch
import polars as pl
from torch.utils.data import Dataset
 
from lunar_text.tokenizer.bpe.wrapper import BPETokenizerWrapper
from lunar_text.training.bpe.masking import Masking
from lunar_text.model.bpe.config import ModelConfig
from lunar_text.utils.tokenizers import REQUIRED_SPECIAL_TOKENS, special_token_id_list

REQUIRED_COLUMNS = {"text"}

class MLMDataset(Dataset):
  def __init__(self, dataset_path: str, wrapper: BPETokenizerWrapper, masking: Masking,
               text_column: str = "text") -> None:
    super().__init__()
    self.wrapper = wrapper
    self.masking = masking
    self.text_column = text_column
    
    df = pl.read_parquet(dataset_path)
    required_columns = REQUIRED_COLUMNS | {text_column}
    missing = required_columns - set(df.columns)
    if missing:
      raise ValueError(
        f"Missing required columns: {sorted(missing)}. "
        f"Found columns: {df.columns}"
      )
    
    self.texts = (
      df
      .with_columns(pl.col(text_column).str.strip_chars().alias(text_column))
      .filter(pl.col(text_column).is_not_null() & (pl.col(text_column) != ""))
      .get_column(text_column)
      .to_list()
    )
    
  def __len__(self):
    return len(self.texts)
    
  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    input_ids                = self.wrapper.encode_retrieval(self.texts[index])
    masked_input_ids, labels = self.masking(input_ids)

    return {
      "input_ids": masked_input_ids,
      "labels": labels,
      "attention_mask": (input_ids != self.wrapper.pad_id).long(),
    }
    
  @staticmethod  
  def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
      key: torch.stack([sample[key] for sample in batch])
      for key in batch[0]
    }
    
def build_dataset(
    dataset_path : str,
    config: ModelConfig,
    tokenizer_path: str,
) -> MLMDataset:
    wrapper = BPETokenizerWrapper(
        tokenizer_path=tokenizer_path,
        max_length=config.max_seq_len,
    )

    if config.vocab_size != wrapper.vocab_size:
        raise ValueError(
            "ModelConfig vocab_size must match tokenizer vocab size. "
            f"Got config.vocab_size={config.vocab_size}, "
            f"tokenizer.vocab_size={wrapper.vocab_size}."
        )

    if config.pad_token_id != wrapper.pad_id:
        raise ValueError(
            "ModelConfig pad_token_id must match tokenizer [PAD] id. "
            f"Got config.pad_token_id={config.pad_token_id}, "
            f"tokenizer.pad_id={wrapper.pad_id}."
        )
 
    masking = Masking(
        vocab_size = wrapper.vocab_size,
        mask_token_id = wrapper.tokenizer.token_to_id("[MASK]"),
        pad_token_id = wrapper.pad_id,
        special_token_ids = special_token_id_list(
            wrapper,
            tokens=REQUIRED_SPECIAL_TOKENS,
            error_context="Tokenizer for MLM masking",
        ),
    )
    return MLMDataset(
      dataset_path=dataset_path,
      wrapper=wrapper,
      masking=masking,
    )
