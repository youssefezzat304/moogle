import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


PatchId = int | str


REQUIRED_COLUMNS = {
    "patch_id",
    "x_coord",
    "y_coord",
    "source_version",
    "prompt_style",
    "text",
}


def validate_columns(df: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {df.columns}"
        )


def validate_split_ratios(train_ratio: float, eval_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1. Got {train_ratio}.")

    if not 0 <= eval_ratio < 1:
        raise ValueError(f"eval_ratio must be between 0 and 1. Got {eval_ratio}.")

    if train_ratio + eval_ratio >= 1:
        raise ValueError(
            "train_ratio + eval_ratio must be less than 1 so a test split remains. "
            f"Got train_ratio={train_ratio}, eval_ratio={eval_ratio}."
        )


def clean_text_rows(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("text").str.strip_chars().alias("text"))
        .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
    )


def get_sorted_patch_ids(df: pl.DataFrame) -> list[PatchId]:
    return (
        df.select("patch_id")
        .unique()
        .sort("patch_id")
        .get_column("patch_id")
        .to_list()
    )


def split_patch_ids(
    patch_ids: list[PatchId],
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[PatchId], list[PatchId], list[PatchId]]:
    validate_split_ratios(train_ratio=train_ratio, eval_ratio=eval_ratio)

    shuffled_ids = patch_ids.copy()

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(shuffled_ids)

    n_patch_ids = len(shuffled_ids)
    train_end = int(train_ratio * n_patch_ids)
    eval_end = train_end + int(eval_ratio * n_patch_ids)

    train_ids = shuffled_ids[:train_end]
    eval_ids = shuffled_ids[train_end:eval_end]
    test_ids = shuffled_ids[eval_end:]

    return train_ids, eval_ids, test_ids


def filter_by_patch_ids(
    df: pl.DataFrame,
    patch_ids: list[PatchId],
) -> pl.DataFrame:
    return df.filter(pl.col("patch_id").is_in(patch_ids))


def validate_no_split_leakage(
    train_ids: list[PatchId],
    eval_ids: list[PatchId],
    test_ids: list[PatchId],
) -> None:
    train_set = set(train_ids)
    eval_set = set(eval_ids)
    test_set = set(test_ids)

    leakage = {
        "train_eval": train_set & eval_set,
        "train_test": train_set & test_set,
        "eval_test": eval_set & test_set,
    }
    leakage = {name: ids for name, ids in leakage.items() if ids}

    if leakage:
        raise ValueError(f"Patch ID leakage detected across splits: {leakage}")


def summarize_splits(
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> dict[str, dict[str, int]]:
    return {
        "train": {
            "rows": train_df.height,
            "patch_ids": train_df.get_column("patch_id").n_unique(),
        },
        "eval": {
            "rows": eval_df.height,
            "patch_ids": eval_df.get_column("patch_id").n_unique(),
        },
        "test": {
            "rows": test_df.height,
            "patch_ids": test_df.get_column("patch_id").n_unique(),
        },
    }


def _value_counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    counts = df.group_by(column).len().sort(column)
    return {
        str(row[column]): row["len"]
        for row in counts.to_dicts()
    }


def build_split_config(
    dataset_path: str,
    saved_path: str,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> dict:
    output_dir = Path(saved_path)
    split_summary = summarize_splits(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(dataset_path)),
        "output_dir": str(output_dir),
        "split_key": "patch_id",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "eval": eval_ratio,
            "test": 1.0 - train_ratio - eval_ratio,
        },
        "files": {
            "train": str(output_dir / "train.parquet"),
            "eval": str(output_dir / "eval.parquet"),
            "test": str(output_dir / "test.parquet"),
            "config": str(output_dir / "config.json"),
        },
        "summary": split_summary,
        "source_version_counts": {
            "train": _value_counts(train_df, "source_version"),
            "eval": _value_counts(eval_df, "source_version"),
            "test": _value_counts(test_df, "source_version"),
        },
        "prompt_style_counts": {
            "train": _value_counts(train_df, "prompt_style"),
            "eval": _value_counts(eval_df, "prompt_style"),
            "test": _value_counts(test_df, "prompt_style"),
        },
    }


def save_split_dataframes(
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    test_df: pl.DataFrame,
    saved_path: str = "data/v1.0",
) -> dict[str, Path]:
    output_dir = Path(saved_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / "train.parquet",
        "eval": output_dir / "eval.parquet",
        "test": output_dir / "test.parquet",
    }

    train_df.write_parquet(paths["train"])
    eval_df.write_parquet(paths["eval"])
    test_df.write_parquet(paths["test"])

    return paths


def save_split_config(config: dict, saved_path: str = "data/v1.0") -> Path:
    output_dir = Path(saved_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config_path


def split_dataset_by_patch_id(
    dataset_path: str,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_parquet(dataset_path)

    validate_columns(df)

    df = clean_text_rows(df)

    patch_ids = get_sorted_patch_ids(df)

    train_ids, eval_ids, test_ids = split_patch_ids(
        patch_ids=patch_ids,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
    )
    validate_no_split_leakage(
        train_ids=train_ids,
        eval_ids=eval_ids,
        test_ids=test_ids,
    )

    train_df = filter_by_patch_ids(df, train_ids)
    eval_df = filter_by_patch_ids(df, eval_ids)
    test_df = filter_by_patch_ids(df, test_ids)

    return train_df, eval_df, test_df


def create_and_save_mlm_splits(
    dataset_path: str,
    saved_path: str = "data/v1.0",
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    train_df, eval_df, test_df = split_dataset_by_patch_id(
        dataset_path=dataset_path,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
    )

    save_split_dataframes(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        saved_path=saved_path,
    )

    config = build_split_config(
        dataset_path=dataset_path,
        saved_path=saved_path,
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
    )
    save_split_config(config=config, saved_path=saved_path)

    return config


def load_dataset(
    dataset_path: str,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return split_dataset_by_patch_id(
        dataset_path=dataset_path,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
    )
