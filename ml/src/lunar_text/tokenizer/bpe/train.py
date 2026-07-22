import polars as pl
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path

def train_bpe_tokenizer(
    dataset_path: str,
    output_path: str, 
    vocab_size: int = 100000,
    text_column: str = "text",
):
    """
    Trains a BPE tokenizer on a Parquet dataset or a raw text file.
    """
    print(f"Loading data from {dataset_path}...")
    path_obj = Path(dataset_path)

    # ── Intelligent Data Loading ──────────────────────────────────────────────
    if path_obj.suffix == ".parquet":
        # Handle Parquet files (extracting the specific column)
        df = pl.read_parquet(dataset_path)
        if text_column in df.columns:
            column_to_use = text_column
        elif "llm_description" in df.columns:
            column_to_use = "llm_description"
        else:
            raise ValueError(
                f"Could not find text column '{text_column}' or legacy "
                f"'llm_description'. Found columns: {df.columns}"
            )
        text_iterator = [
            text
            for text in (
                df
                .select(pl.col(column_to_use).cast(pl.Utf8).str.strip_chars())
                .get_column(column_to_use)
                .drop_nulls()
                .to_list()
            )
            if text != ""
        ]
        
    elif path_obj.suffix == ".txt":
        # Handle raw Text files (reading line by line)
        with open(dataset_path, "r", encoding="utf-8") as f:
            # .strip() removes trailing newlines so the tokenizer doesn't learn garbage whitespace
            text_iterator = [line.strip() for line in f if line.strip()]
            
    else:
        raise ValueError(f"Unsupported file format: '{path_obj.suffix}'. Expected .parquet or .txt")

    # ── Tokenizer Training ────────────────────────────────────────────────────
    print(f"Loaded {len(text_iterator)} descriptions. Initializing tokenizer...")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=[
            "[PAD]",
            "[SOS]",
            "[EOS]",
            "[MASK]",
            "[RETRIEVAL]",
            "[UNK]",
        ]
    )

    print("Training BPE model (this is usually very fast)...")
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save(output_path)
    print(f"Success! Tokenizer saved to: {output_path}")
