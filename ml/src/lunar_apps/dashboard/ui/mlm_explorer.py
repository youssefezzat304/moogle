from __future__ import annotations

import html
from pathlib import Path

import polars as pl
import streamlit as st
import torch

from lunar_text.tokenizer.bpe.wrapper import BPETokenizerWrapper
from lunar_text.training.bpe.masking import Masking
from lunar_text.utils.tokenizers import load_tokenizer, special_token_id_list


TOKENIZER_ROOT = Path("artifacts/tokenizers")
REQUIRED_MLM_TOKENS = ("[PAD]", "[SOS]", "[EOS]", "[MASK]")
OPTIONAL_SPECIAL_TOKENS = ("[RETRIEVAL]", "[UNK]")
IGNORED_LABEL = -100

STATE_LABELS = {
    "mask": "masked",
    "random": "random",
    "unchanged": "unchanged target",
    "visible": "visible",
    "special": "special",
    "padding": "padding",
}

STATE_COLOURS = {
    "mask": "#6D597A",
    "random": "#B56576",
    "unchanged": "#2A9D8F",
    "visible": "#457B9D",
    "special": "#6C757D",
    "padding": "#E9ECEF",
}

STATE_TEXT_COLOURS = {
    "mask": "#FFFFFF",
    "random": "#FFFFFF",
    "unchanged": "#FFFFFF",
    "visible": "#FFFFFF",
    "special": "#FFFFFF",
    "padding": "#495057",
}


@st.cache_data
def _discover_mlm_tokenizer_versions(root: str = str(TOKENIZER_ROOT)) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    versions: list[str] = []
    for tokenizer_path in root_path.rglob("tokenizer.json"):
        version = tokenizer_path.parent.relative_to(root_path).as_posix()
        try:
            tokenizer = load_tokenizer(tokenizer_path)
        except Exception:
            continue

        if all(tokenizer.token_to_id(token) is not None for token in REQUIRED_MLM_TOKENS):
            versions.append(version)

    return sorted(versions, key=_version_sort_key)


def _version_sort_key(version: str) -> tuple[str, int, list[int], str]:
    parts = version.split("/")
    family = parts[0]
    label = parts[-1]
    numbers = [int(part) for part in label.lstrip("v").split(".") if part.isdigit()]
    return family, len(parts), numbers, version


@st.cache_resource
def _load_bpe_wrapper(version: str, max_seq_len: int) -> BPETokenizerWrapper:
    tokenizer_path = TOKENIZER_ROOT / version / "tokenizer.json"
    return BPETokenizerWrapper(
        tokenizer_path=str(tokenizer_path),
        max_length=max_seq_len,
    )


def _build_masking(wrapper: BPETokenizerWrapper, mask_probability: float) -> Masking:
    mask_token_id = wrapper.tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        raise ValueError("Selected tokenizer does not contain [MASK].")

    return Masking(
        vocab_size=wrapper.vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=wrapper.pad_id,
        special_token_ids=special_token_id_list(
            wrapper,
            tokens=REQUIRED_MLM_TOKENS + OPTIONAL_SPECIAL_TOKENS,
            require_all=False,
        ),
        mask_probability=mask_probability,
    )


def _is_usable_caption(caption: str | None) -> bool:
    if not caption:
        return False

    known_error_prefixes = (
        "Text pipeline",
        "No descriptions",
        "Could not locate",
    )
    return not caption.startswith(known_error_prefixes)


def _filter_patch(dataframe: pl.DataFrame, patch_id: int | None) -> pl.DataFrame:
    if patch_id is None:
        return dataframe

    if "patch_id" in dataframe.columns:
        return dataframe.filter(pl.col("patch_id") == patch_id)

    if "patch_number" in dataframe.columns:
        return dataframe.filter(pl.col("patch_number") == patch_id)

    return dataframe


def _collect_texts(
    dataframe: pl.DataFrame,
    selected_style: str,
    patch_id: int | None = None,
    limit: int = 32,
) -> list[str]:
    if dataframe.is_empty():
        return []

    frame = _filter_patch(dataframe, patch_id)
    if frame.is_empty():
        frame = dataframe

    if "text" in frame.columns:
        text_frame = frame
        if selected_style != "Error" and "prompt_style" in frame.columns:
            style_frame = frame.filter(pl.col("prompt_style") == selected_style)
            if not style_frame.is_empty():
                text_frame = style_frame

        return _nonnull_text_values(text_frame, "text", limit)

    caption_columns = [column for column in frame.columns if column.startswith("caption_")]
    preferred_column = f"caption_{selected_style}"
    if preferred_column in caption_columns:
        return _nonnull_text_values(frame, preferred_column, limit)

    for column in caption_columns:
        values = _nonnull_text_values(frame, column, limit)
        if values:
            return values

    if "llm_description" in frame.columns:
        return _nonnull_text_values(frame, "llm_description", limit)

    return []


def _nonnull_text_values(frame: pl.DataFrame, column: str, limit: int) -> list[str]:
    return (
        frame.select(pl.col(column).cast(pl.Utf8).str.strip_chars().alias(column))
        .filter(pl.col(column).is_not_null() & (pl.col(column) != ""))
        .head(limit)
        .get_column(column)
        .to_list()
    )


def _default_text(
    caption: str,
    dataframe: pl.DataFrame,
    selected_style: str,
    patch_id: int | None,
) -> str:
    if _is_usable_caption(caption):
        return caption

    patch_texts = _collect_texts(
        dataframe=dataframe,
        selected_style=selected_style,
        patch_id=patch_id,
        limit=1,
    )
    if patch_texts:
        return patch_texts[0]

    corpus_texts = _collect_texts(
        dataframe=dataframe,
        selected_style=selected_style,
        patch_id=None,
        limit=1,
    )
    return corpus_texts[0] if corpus_texts else ""


def _mask_input(
    wrapper: BPETokenizerWrapper,
    masking: Masking,
    text: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = wrapper.encode(text)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        masked_input_ids, labels = masking(input_ids)
    return input_ids, masked_input_ids, labels


def _token_for_id(wrapper: BPETokenizerWrapper, token_id: int) -> str:
    token = wrapper.tokenizer.id_to_token(token_id)
    return token if token is not None else f"<id:{token_id}>"


def _display_token(token: str) -> str:
    token = token.replace("▁", " ")
    return html.escape(token)


def _classify_token(
    original_id: int,
    masked_id: int,
    label_id: int,
    wrapper: BPETokenizerWrapper,
) -> str:
    if original_id == wrapper.pad_id:
        return "padding"

    if label_id == IGNORED_LABEL:
        if original_id in set(
            special_token_id_list(
                wrapper,
                tokens=REQUIRED_MLM_TOKENS + OPTIONAL_SPECIAL_TOKENS,
                require_all=False,
            )
        ):
            return "special"
        return "visible"

    mask_token_id = wrapper.tokenizer.token_to_id("[MASK]")
    if masked_id == mask_token_id:
        return "mask"

    if masked_id == original_id:
        return "unchanged"

    return "random"


def _build_token_rows(
    wrapper: BPETokenizerWrapper,
    original_ids: torch.Tensor,
    masked_ids: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict[str, str | int | bool]]:
    rows: list[dict[str, str | int | bool]] = []
    for position, (original_id, masked_id, label_id) in enumerate(
        zip(original_ids.tolist(), masked_ids.tolist(), labels.tolist(), strict=True)
    ):
        state = _classify_token(
            original_id=original_id,
            masked_id=masked_id,
            label_id=label_id,
            wrapper=wrapper,
        )
        label_token = (
            "ignored"
            if label_id == IGNORED_LABEL
            else _token_for_id(wrapper, int(label_id))
        )

        rows.append(
            {
                "position": position,
                "original_id": int(original_id),
                "original_token": _token_for_id(wrapper, int(original_id)),
                "model_input_id": int(masked_id),
                "model_input_token": _token_for_id(wrapper, int(masked_id)),
                "label_id": "ignored" if label_id == IGNORED_LABEL else int(label_id),
                "label_token": label_token,
                "loss_active": label_id != IGNORED_LABEL,
                "state": STATE_LABELS[state],
                "_state_key": state,
            }
        )

    return rows


def _render_token_timeline(rows: list[dict[str, str | int | bool]]) -> None:
    pills: list[str] = []
    for row in rows:
        state = str(row["_state_key"])
        bg = STATE_COLOURS[state]
        fg = STATE_TEXT_COLOURS[state]
        border = "1px solid #CED4DA" if state == "padding" else "1px solid transparent"
        token = _display_token(str(row["model_input_token"]))
        title = html.escape(
            f"pos={row['position']} id={row['model_input_id']} state={row['state']}"
        )
        pills.append(
            f'<span title="{title}" style="'
            f"background:{bg};color:{fg};border:{border};padding:3px 7px;"
            "margin:2px 1px;border-radius:8px;font-size:0.78rem;"
            "display:inline-block;font-family:monospace;line-height:1.4;"
            '">'
            f"{token}</span>"
        )

    st.markdown(
        '<div style="line-height:2.2;padding:4px 0;overflow-wrap:anywhere;">'
        + "".join(pills)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_state_legend() -> None:
    chips: list[str] = []
    for state, label in STATE_LABELS.items():
        bg = STATE_COLOURS[state]
        fg = STATE_TEXT_COLOURS[state]
        border = "1px solid #CED4DA" if state == "padding" else "1px solid transparent"
        chips.append(
            f'<span style="background:{bg};color:{fg};border:{border};'
            "padding:3px 8px;margin:2px 3px;border-radius:8px;"
            f'font-size:0.78rem;display:inline-block;">{html.escape(label)}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def render_mlm_explorer(
    caption: str,
    dataframe: pl.DataFrame,
    selected_style: str,
    patch_id: int | None = None,
) -> None:
    st.header("MLM Process Explorer")

    versions = _discover_mlm_tokenizer_versions()
    if not versions:
        st.warning("No MLM-compatible tokenizers found under `artifacts/tokenizers/`.")
        return

    default_version = "bpe/v4.0" if "bpe/v4.0" in versions else versions[-1]
    default_caption = _default_text(
        caption=caption,
        dataframe=dataframe,
        selected_style=selected_style,
        patch_id=patch_id,
    )

    with st.expander("MLM Settings", expanded=True):
        control_cols = st.columns(3)
        with control_cols[0]:
            tokenizer_version = st.selectbox(
                "MLM tokenizer",
                options=versions,
                index=versions.index(default_version),
            )
            max_seq_len = st.slider(
                "Sequence length",
                min_value=16,
                max_value=400,
                value=64,
                step=16,
            )
        with control_cols[1]:
            mask_probability = st.slider(
                "Mask probability",
                min_value=0.0,
                max_value=0.5,
                value=0.15,
                step=0.01,
            )
            seed = st.number_input(
                "Mask seed",
                min_value=0,
                max_value=1_000_000,
                value=42,
                step=1,
            )
        with control_cols[2]:
            st.metric("Available texts", f"{dataframe.height:,}")

        input_text = st.text_area(
            "Input caption",
            value=default_caption,
            height=130,
        )

    if not input_text.strip():
        st.info("No caption text available for MLM inspection.")
        return

    try:
        wrapper = _load_bpe_wrapper(tokenizer_version, max_seq_len)
        masking = _build_masking(wrapper, mask_probability)
        original_ids, masked_ids, labels = _mask_input(
            wrapper=wrapper,
            masking=masking,
            text=input_text,
            seed=int(seed),
        )
    except Exception as exc:
        st.error(f"Could not build MLM sample: {exc}")
        return

    rows = _build_token_rows(
        wrapper=wrapper,
        original_ids=original_ids,
        masked_ids=masked_ids,
        labels=labels,
    )

    st.subheader("Token Timeline")
    _render_state_legend()
    _render_token_timeline(rows)
