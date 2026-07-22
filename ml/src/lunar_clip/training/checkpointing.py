from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint


def build_clip_checkpoint_callback(
    output_dir: str | Path,
    monitor: str,
    save_top_k: int,
) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename="best",
        monitor=monitor,
        mode="min",
        save_top_k=save_top_k,
        save_last=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
