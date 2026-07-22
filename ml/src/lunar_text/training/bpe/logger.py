from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "run_id",
    "timestamp",
    "elapsed_seconds",
    "epoch",
    "step",
    "split",
    "loss",
    "masked_accuracy",
    "masked_perplexity",
    "masked_tokens",
    "levenshtein_distance",
    "learning_rate",
    "grad_norm",
]


class MLMRunLogger:
    def __init__(
        self,
        output_root: str = "results/mlm/runs",
        run_name: str = "smoke",
    ) -> None:
        created_at = datetime.now(timezone.utc)
        safe_name = _safe_run_name(run_name)
        base_run_id = f"{created_at.strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        self.created_at = created_at.isoformat()

        output_dir = Path(output_root)
        self.run_id = _unique_run_id(output_dir, base_run_id)
        self.run_dir = output_dir / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.json"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=False)
        self._write_metrics_header()

    def save_config(self, config: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "run_dir": str(self.run_dir),
            **_json_safe(config),
        }
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        row = {field: "" for field in METRIC_FIELDS}
        row.update(
            {
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        row.update(metrics)

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field, "")
                    for field in METRIC_FIELDS
                }
            )

    def _write_metrics_header(self) -> None:
        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            writer.writeheader()


def _safe_run_name(run_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name.strip())
    return cleaned.strip("-") or "run"


def _unique_run_id(output_root: Path, base_run_id: str) -> str:
    run_id = base_run_id
    suffix = 2
    while (output_root / run_id).exists():
        run_id = f"{base_run_id}_{suffix}"
        suffix += 1
    return run_id


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value
