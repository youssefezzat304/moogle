from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torch.nn.functional as F

from lunar_text.model.bpe.config import ModelConfig
from lunar_text.model.bpe.model import BPELunarMLM
from lunar_text.tokenizer.bpe.wrapper import BPETokenizerWrapper
from lunar_text.training.bpe.masking import Masking
from lunar_text.utils.checkpoints import (
    checkpoint_value,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)
from lunar_text.utils.tokenizers import REQUIRED_SPECIAL_TOKENS, special_token_id_list


IGNORE_INDEX = -100


@dataclass(frozen=True)
class LoadedMLM:
    model: BPELunarMLM
    wrapper: BPETokenizerWrapper
    checkpoint: dict[str, Any]
    device: str


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []

    checkpoints = [
        *checkpoints_dir.glob("*.pt"),
        *checkpoints_dir.glob("*.ckpt"),
    ]
    return sorted(
        checkpoints,
        key=lambda path: (
            _is_final_checkpoint(path),
            _checkpoint_step(path) if _checkpoint_step(path) is not None else 10**12,
            path.name,
        ),
    )


def checkpoint_label(path: Path) -> str:
    if _is_final_checkpoint(path):
        return "final"
    step = _checkpoint_step(path)
    return f"step {step:,}" if step is not None else path.stem


def default_checkpoint_index(checkpoints: list[Path]) -> int:
    if not checkpoints:
        return 0

    for index, path in enumerate(checkpoints):
        if _is_final_checkpoint(path):
            return index

    return len(checkpoints) - 1


@st.cache_resource(show_spinner=False)
def load_mlm_checkpoint(checkpoint_path: str, device: str = "cpu") -> LoadedMLM:
    resolved_device = _resolve_device(device)
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=resolved_device)
    if checkpoint is None:
        raise ValueError(f"Checkpoint could not be loaded: {checkpoint_path}")
    config = ModelConfig(**checkpoint_value(checkpoint, "model_config"))
    model = BPELunarMLM(config)
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint))
    model.to(resolved_device)
    model.eval()

    tokenizer_path = checkpoint_value(checkpoint, "tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Checkpoint does not contain tokenizer_path metadata.")

    wrapper = BPETokenizerWrapper(
        tokenizer_path=tokenizer_path,
        max_length=config.max_seq_len,
    )

    checkpoint.setdefault("model_config", checkpoint_value(checkpoint, "model_config"))
    checkpoint.setdefault(
        "training_config", checkpoint_value(checkpoint, "training_config") or {}
    )
    checkpoint.setdefault("tokenizer_path", tokenizer_path)

    return LoadedMLM(
        model=model,
        wrapper=wrapper,
        checkpoint=checkpoint,
        device=str(resolved_device),
    )


def analyze_masked_text(
    loaded: LoadedMLM,
    text: str,
    mask_probability: float,
    seed: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    masking = _build_masking(loaded.wrapper, mask_probability)
    original_ids = loaded.wrapper.encode(text)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        masked_ids, labels = masking(original_ids)

    input_batch = masked_ids.unsqueeze(0).to(loaded.device)
    with torch.no_grad():
        logits = loaded.model(input_batch)[0].detach().cpu()

    probabilities = F.softmax(logits, dim=-1)
    rows = _prediction_rows(
        wrapper=loaded.wrapper,
        original_ids=original_ids,
        masked_ids=masked_ids,
        labels=labels,
        probabilities=probabilities,
        top_k=top_k,
    )
    return rows, logits


def render_correctness_timeline(rows: list[dict[str, Any]], top_k: int) -> None:
    chips = []
    for row in rows:
        state = _timeline_state(row)
        bg, fg = _timeline_colours(state)
        token = display_token(str(row["model_input_token"]))
        tooltip = html.escape(
            " | ".join(
                [
                    f"pos={row['position']}",
                    f"input={row['model_input_token']}",
                    f"target={row['target_token']}",
                    f"top={row['top_prediction']}",
                    f"p={row['top_probability']:.3f}",
                    f"rank={row['target_rank']}",
                    f"top{top_k}={row['target_in_top_k']}",
                ]
            )
        )
        chips.append(
            f'<span title="{tooltip}" style="background:{bg};color:{fg};'
            "padding:3px 7px;margin:2px 1px;border-radius:8px;"
            "font-size:0.78rem;display:inline-block;font-family:monospace;"
            f'line-height:1.4;">{token}</span>'
        )

    st.markdown(
        '<div style="line-height:2.2;padding:4px 0;overflow-wrap:anywhere;">'
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


def token_for_id(wrapper: BPETokenizerWrapper, token_id: int) -> str:
    token = wrapper.tokenizer.id_to_token(int(token_id))
    return token if token is not None else f"<id:{token_id}>"


def display_token(token: str) -> str:
    return html.escape(token.replace("▁", " "))


def _build_masking(wrapper: BPETokenizerWrapper, mask_probability: float) -> Masking:
    mask_token_id = wrapper.tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        raise ValueError("Tokenizer must contain [MASK].")

    return Masking(
        vocab_size=wrapper.vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=wrapper.pad_id,
        special_token_ids=special_token_id_list(
            wrapper,
            tokens=REQUIRED_SPECIAL_TOKENS,
            require_all=False,
        ),
        mask_probability=mask_probability,
    )


def _prediction_rows(
    wrapper: BPETokenizerWrapper,
    original_ids: torch.Tensor,
    masked_ids: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top_values, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

    for position, (original_id, masked_id, label_id) in enumerate(
        zip(original_ids.tolist(), masked_ids.tolist(), labels.tolist(), strict=True)
    ):
        active = int(label_id) != IGNORE_INDEX
        target_rank = None
        target_probability = None
        top_prediction_id = int(top_indices[position, 0].item())
        top_prediction = token_for_id(wrapper, top_prediction_id)
        top_probability = float(top_values[position, 0].item())
        target_in_top_k = False

        if active:
            token_probabilities = probabilities[position]
            target_probability = float(token_probabilities[int(label_id)].item())
            target_rank = (
                int((token_probabilities > target_probability).sum().item()) + 1
            )
            target_in_top_k = int(label_id) in top_indices[position].tolist()

        rows.append(
            {
                "position": position,
                "original_id": int(original_id),
                "original_token": token_for_id(wrapper, int(original_id)),
                "model_input_id": int(masked_id),
                "model_input_token": token_for_id(wrapper, int(masked_id)),
                "target_id": None if not active else int(label_id),
                "target_token": ""
                if not active
                else token_for_id(wrapper, int(label_id)),
                "loss_active": active,
                "top_prediction_id": top_prediction_id,
                "top_prediction": top_prediction,
                "top_probability": top_probability,
                "target_probability": target_probability,
                "target_rank": target_rank,
                "target_in_top_k": target_in_top_k,
                "correct_top1": active and top_prediction_id == int(label_id),
                "top_predictions": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token": token_for_id(wrapper, int(token_id)),
                        "probability": float(probability),
                    }
                    for rank, (token_id, probability) in enumerate(
                        zip(
                            top_indices[position].tolist(),
                            top_values[position].tolist(),
                            strict=True,
                        )
                    )
                ],
            }
        )
    return rows


def _checkpoint_step(path: Path) -> int | None:
    match = re.search(r"step[_=](\d+)", path.name)
    return int(match.group(1)) if match else None


def _is_final_checkpoint(path: Path) -> bool:
    return path.name in {"final.pt", "final.ckpt", "last.ckpt"}


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _timeline_state(row: dict[str, Any]) -> str:
    if not row["loss_active"]:
        return "inactive"
    if row["correct_top1"]:
        return "correct"
    if row["target_in_top_k"]:
        return "topk"
    return "wrong"


def _timeline_colours(state: str) -> tuple[str, str]:
    if state == "correct":
        return "#2A9D8F", "#FFFFFF"
    if state == "topk":
        return "#E9C46A", "#212529"
    if state == "wrong":
        return "#C1121F", "#FFFFFF"
    return "#E9ECEF", "#495057"
