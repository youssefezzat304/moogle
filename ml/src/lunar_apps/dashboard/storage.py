import pathlib
from datetime import datetime

import polars as pl

from lunar_apps.dashboard.config import CONFIG


def build_output_path(output_dir: str | pathlib.Path) -> pathlib.Path:
    """
    Build the fixed parquet output path based on patch_size and stride.

    Output format:
    results_ps{patch_size}_s{stride}.parquet
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size = CONFIG["hyperparameters"]["patch_size"]
    stride = CONFIG["hyperparameters"]["stride"]

    return output_dir / f"results_ps{patch_size}_s{stride}.parquet"


def last_completed_patch(output_dir: str | pathlib.Path) -> int:
    """
    Read the existing parquet file (if any) and return the highest
    patch_number found.

    Returns:
        -1 if no file exists (fresh run)
    """
    path = build_output_path(output_dir)

    if not path.exists():
        return -1

    df = pl.read_parquet(path, columns=["patch_number"])
    max_patch = df["patch_number"].max()

    if max_patch is not None:
        print(f"  ↳ Resuming from patch {max_patch + 1}")
        return max_patch

    return -1


def _write(data: list[dict], path: pathlib.Path) -> pathlib.Path:
    """
    Append data to the parquet file if it exists,
    otherwise create it.
    """
    df_new = pl.from_dicts(data)

    if path.exists():
        df_existing = pl.read_parquet(path)
        df_new = pl.concat([df_existing, df_new])

    df_new.write_parquet(path)
    print(f"  ✓ {len(data)} records → {path}")

    return path


def checkpoint(
    buffer: list[dict],
    result: dict,
    every_n: int,
    output_dir: str | pathlib.Path,
) -> None:
    """
    Append a result to the in-memory buffer.

    When the buffer reaches every_n records, stamp those records
    with the same batch timestamp and flush them to parquet.

    Call this inside the inference loop after each sample.
    """
    buffer.append(result)

    if len(buffer) % every_n == 0:
        batch_ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        for record in buffer[-every_n:]:
            record["saved_at"] = batch_ts

        _write(
            data=buffer[-every_n:],
            path=build_output_path(output_dir),
        )


def flush(
    buffer: list[dict],
    output_dir: str | pathlib.Path,
) -> pathlib.Path | None:
    """
    Write any remaining buffered records that did not hit
    a checkpoint boundary.

    Stamps them with the current timestamp.

    Call this once at the end of the run.

    Returns:
        Path written, or None if the buffer was empty.
    """
    remainder = [record for record in buffer if "saved_at" not in record]

    if not remainder:
        buffer.clear()
        return None

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for record in remainder:
        record["saved_at"] = batch_ts

    path = _write(
        data=remainder,
        path=build_output_path(output_dir),
    )

    buffer.clear()
    return path
