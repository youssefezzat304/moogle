from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
import torch

from lunar_apps.dashboard.ui.mlm_model_analysis import (
    analyze_masked_text,
    checkpoint_label,
    default_checkpoint_index,
    discover_checkpoints,
    load_mlm_checkpoint,
    render_correctness_timeline,
)


RUNS_ROOT = "results/mlm/runs"


def render_mlm_training_dashboard(runs_root: str = RUNS_ROOT) -> None:
    st.header("MLM Training Runs")

    runs = _discover_runs(runs_root)
    if not runs:
        st.info(f"No MLM training runs found under `{runs_root}`.")
        return

    selected_run = st.selectbox(
        "Training run",
        options=runs,
        format_func=lambda path: path.name,
    )
    if selected_run is None:
        return

    config = _render_run_summary(selected_run=selected_run, runs_root=runs_root)
    checkpoints = discover_checkpoints(selected_run)

    st.divider()
    _render_prediction_explorer(checkpoints=checkpoints, config=config)


def _discover_runs(runs_root: str) -> list[Path]:
    root = Path(runs_root)
    if not root.exists():
        return []

    runs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and ((path / "config.json").exists() or (path / "checkpoints").exists())
    ]
    return sorted(runs, key=lambda path: path.name, reverse=True)


def _render_run_summary(selected_run: Path, runs_root: str) -> dict[str, Any]:
    config = _load_json(selected_run / "config.json")

    st.caption(f"Run directory: `{selected_run}`")
    tensorboard_log_dir = config.get("tensorboard_log_dir") or str(selected_run)
    st.code(
        f"uv run main.py mlm export-tensorboard --runs-root {runs_root}\n"
        f"uv run tensorboard --logdir {runs_root}",
        language="bash",
    )
    st.caption(f"TensorBoard events: `{tensorboard_log_dir}`")

    if not config:
        return {}

    training = config.get("training", {})
    model = config.get("model", {})
    summary_cols = st.columns(4)
    summary_cols[0].metric("Max seq len", training.get("max_seq_len", ""))
    summary_cols[1].metric("Batch size", training.get("batch_size", ""))
    summary_cols[2].metric("Learning rate", training.get("learning_rate", ""))
    summary_cols[3].metric("Vocab size", model.get("vocab_size", ""))

    with st.expander("Run config", expanded=False):
        st.json(config)

    return config


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _render_prediction_explorer(
    checkpoints: list[Path],
    config: dict[str, Any],
) -> None:
    st.subheader("Prediction Explorer")
    if not checkpoints:
        st.info("No checkpoints found for this run.")
        return

    checkpoint, device = _checkpoint_controls(checkpoints=checkpoints)

    try:
        loaded = load_mlm_checkpoint(str(checkpoint), device)
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return

    default_text = _default_eval_text(loaded.checkpoint) or _default_eval_text(config)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        mask_probability = st.slider(
            "Mask probability",
            min_value=0.01,
            max_value=0.50,
            value=0.15,
            step=0.01,
            key="prediction_mask_probability",
        )
    with col_b:
        seed = st.number_input(
            "Mask seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
            key="prediction_seed",
        )
    with col_c:
        top_k = st.slider(
            "Top-k",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="prediction_top_k",
        )

    text = st.text_area("Input text", value=default_text, height=140)
    if not text.strip():
        st.info("Enter text to inspect predictions.")
        return

    try:
        rows, logits = analyze_masked_text(
            loaded=loaded,
            text=text,
            mask_probability=float(mask_probability),
            seed=int(seed),
            top_k=int(top_k),
        )
    except Exception as exc:
        st.error(f"Could not run prediction analysis: {exc}")
        return

    st.caption(f"Logits shape: `{tuple(logits.unsqueeze(0).shape)}`")
    render_correctness_timeline(rows, top_k=int(top_k))

    active_rows = [row for row in rows if row["loss_active"]]
    if not active_rows:
        st.warning(
            "No active masked positions. Increase mask probability or change seed."
        )
        return

    correct = sum(1 for row in active_rows if row["correct_top1"])
    in_top_k = sum(1 for row in active_rows if row["target_in_top_k"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Masked tokens", len(active_rows))
    metric_cols[1].metric("Top-1 correct", correct, f"{correct / len(active_rows):.1%}")
    metric_cols[2].metric(
        f"Top-{top_k} contains target",
        in_top_k,
        f"{in_top_k / len(active_rows):.1%}",
    )
    metric_cols[3].metric("Checkpoint", checkpoint_label(checkpoint))

    table = pl.DataFrame(
        [
            {
                "position": row["position"],
                "input": row["model_input_token"],
                "target": row["target_token"],
                "prediction": row["top_prediction"],
                "confidence": row["top_probability"],
                "target_probability": row["target_probability"],
                "target_rank": row["target_rank"],
                "correct_top1": row["correct_top1"],
                f"in_top_{top_k}": row["target_in_top_k"],
            }
            for row in active_rows
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    selected_position = st.selectbox(
        "Top-k bars for masked position",
        options=[row["position"] for row in active_rows],
        format_func=lambda position: f"position {position}",
    )
    selected_row = next(
        row for row in active_rows if row["position"] == selected_position
    )
    top_predictions = pl.DataFrame(selected_row["top_predictions"])
    st.bar_chart(
        top_predictions.to_pandas(),
        x="token",
        y="probability",
        use_container_width=True,
    )


def _checkpoint_controls(checkpoints: list[Path]) -> tuple[Path, str]:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        checkpoint = st.selectbox(
            "Checkpoint",
            options=checkpoints,
            index=default_checkpoint_index(checkpoints),
            format_func=checkpoint_label,
            key="prediction_checkpoint",
        )
    with col_b:
        device = _device_selector()

    return checkpoint, device


def _device_selector() -> str:
    options = ["cpu"]
    if torch.cuda.is_available():
        options.append("cuda")
    return st.selectbox(
        "Inference device",
        options=options,
        index=0,
        key="prediction_device",
    )


def _default_eval_text(metadata: dict[str, Any]) -> str:
    eval_path = metadata.get("training_config", {}).get("eval_path")
    if not eval_path:
        eval_path = metadata.get("training", {}).get("eval_path")
    if not eval_path or not Path(eval_path).exists():
        return ""

    try:
        frame = pl.read_parquet(eval_path, columns=["text"], n_rows=1)
    except Exception:
        return ""

    if frame.is_empty():
        return ""
    return str(frame.get_column("text").item())
