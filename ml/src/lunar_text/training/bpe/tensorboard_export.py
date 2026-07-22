from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


METRIC_COLUMNS = (
    "loss",
    "masked_accuracy",
    "masked_perplexity",
    "masked_tokens",
    "levenshtein_distance",
    "learning_rate",
    "grad_norm",
)


def export_metrics_csv_to_tensorboard(
    runs_root: str = "results/mlm/runs",
    force: bool = False,
) -> list[Path]:
    root = Path(runs_root)
    if not root.exists():
        return []

    exported: list[Path] = []
    for metrics_path in sorted(root.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        if not force and _has_event_file(run_dir):
            continue
        if _export_run(metrics_path=metrics_path, log_dir=run_dir):
            exported.append(run_dir)

    return exported


def log_metrics_to_tensorboard(
    writer: SummaryWriter,
    metrics: dict[str, object],
) -> None:
    split = str(metrics.get("split") or "metrics")
    step = _as_int(metrics.get("step"))
    if step is None:
        return

    for column in METRIC_COLUMNS:
        value = _as_float(metrics.get(column))
        if value is None:
            continue
        writer.add_scalar(f"{split}/{column}", value, step)


def _export_run(metrics_path: Path, log_dir: Path) -> bool:
    rows_written = 0
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        with SummaryWriter(log_dir=str(log_dir)) as writer:
            for row in reader:
                before = rows_written
                log_metrics_to_tensorboard(writer, row)
                if _as_int(row.get("step")) is not None:
                    rows_written += 1
                if rows_written == before:
                    continue
            writer.flush()

    return rows_written > 0


def _has_event_file(run_dir: Path) -> bool:
    return any(run_dir.rglob("events.out.tfevents*"))


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None
