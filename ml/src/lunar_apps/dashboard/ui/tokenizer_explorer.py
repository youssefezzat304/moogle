import streamlit as st
import polars as pl
from tokenizers import Tokenizer
import csv
from tokenizers.models import Unigram

from lunar_data.loaders import load_tokenizer, get_corpus_texts, discover_tokenizer_versions
from lunar_apps.dashboard.ui.styles import _TOKEN_COLOURS, _PROBE_DEFAULTS
from lunar_text.tokenizer.eval.metrics import (
    vocab_overlap,
    fertility,
    unk_rate,
    fragmentation_check,
    token_frequency,
)

def render_tokenizer_explorer(caption: str, corpus_version: str, selected_style: str):
    st.header("Tokenizer Explorer")

    # ── Tokenizer Configuration Controls ──────────────────────────────────────────
    tok_versions = discover_tokenizer_versions()

    if not tok_versions:
        st.warning("No tokenizers found under `artifacts/tokenizers/`.")
        return

    with st.expander("⚙️ Tokenizer Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_tokenizers = st.multiselect(
                "Tokenizer version(s)",
                options=tok_versions,
                default=[tok_versions[-1]],
                help="Format: {algorithm}/{version}. Pick one to inspect, two or more to compare side-by-side.",
            )
            uploaded_file = st.file_uploader("Upload tokenizer", type=["json", "csv"])
            
        with col2:
            probe_input = st.text_area(
                "Fragmentation probe terms",
                value=_PROBE_DEFAULTS,
                height=110,
                help="Terms to check for fragmentation (separated by spaces or commas).",
            )
            probe_terms = [t for t in probe_input.replace(",", " ").split() if t]
            top_n = st.slider("Top-N tokens", min_value=5, max_value=50, value=15)

    if not selected_tokenizers and not uploaded_file:
        st.info("Select one or more tokenizer versions in the settings above, or upload one to begin.")
        return

    # Load tokenizers
    tokenizers: dict[str, Tokenizer] = {}
    for v in selected_tokenizers:
        try:
            tokenizers[v] = load_tokenizer(v)
        except Exception as e:
            st.error(f"Could not load tokenizer **{v}**: {e}")

    # Process uploaded tokenizer in memory
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue().decode("utf-8")
            
            if file_name.endswith(".json"):
                # The tokenizers library strictly validates the JSON structure automatically
                uploaded_tok = Tokenizer.from_str(file_bytes)
                tokenizers[f"Uploaded: {file_name}"] = uploaded_tok
                
            elif file_name.endswith(".csv"):
                # Parse CSV directly from memory
                reader = csv.reader(file_bytes.splitlines())
                next(reader, None)  # Skip header
                vocab_scores = []
                for row in reader:
                    if len(row) >= 2:
                        vocab_scores.append((row[0], float(row[1])))
                
                unigram_model = Unigram(vocab_scores)
                uploaded_tok = Tokenizer(unigram_model)
                tokenizers[f"Uploaded: {file_name}"] = uploaded_tok
                
        except Exception as e:
            st.error(f"Failed to load uploaded tokenizer `{uploaded_file.name}`: {e}")

    if not tokenizers:
        return

    # Load corpus directly using the dataset version and style from the sidebar/UI
    corpus_texts: list[str] = []
    if corpus_version and selected_style != "Error":
        corpus_texts = get_corpus_texts(corpus_version, selected_style)
        if corpus_texts:
            st.caption(
                f"Metrics computed over **{len(corpus_texts):,}** '{selected_style}' descriptions "
                f"from currently selected dataset: **{corpus_version}**."
            )
        else:
            st.warning(
                f"No parquet found for corpus **{corpus_version}** or style **{selected_style}**. "
                "Fertility, UNK rate, and frequency metrics unavailable."
            )

    # ── 1. Token breakdown of the current patch description ───────────────────────
    st.subheader("1 · Token Breakdown — Current Description")
    st.caption(
        "How each tokenizer splits the caption into tokens. "
        "Hover a pill to see its token ID. Colors cycle per token position."
    )

    if caption.startswith("Text pipeline"):
        st.warning("No caption available for this patch. Navigate to a processed patch.")
    else:
        for version, tok in tokenizers.items():
            encoded  = tok.encode(caption)
            tokens   = encoded.tokens
            ids      = encoded.ids
            n_words  = len(caption.split())
            f_score  = round(len(tokens) / n_words, 3) if n_words else 0

            st.markdown(
                f"**{version}** &nbsp;·&nbsp; "
                f"`{len(tokens)}` tokens &nbsp;·&nbsp; "
                f"`{n_words}` words &nbsp;·&nbsp; "
                f"fertility `{f_score}`"
            )

            pills = []
            for i, (token, tid) in enumerate(zip(tokens, ids)):
                colour  = _TOKEN_COLOURS[i % len(_TOKEN_COLOURS)]
                display = token.replace("▁", " ").replace("<", "&lt;").replace(">", "&gt;")
                pills.append(
                    f'<span title="id={tid}" style="'
                    f'background:{colour};color:#fff;padding:2px 8px;margin:2px 1px;'
                    f'border-radius:12px;font-size:0.80rem;display:inline-block;'
                    f'font-family:monospace;cursor:default;">'
                    f'{display}</span>'
                )

            st.markdown(
                '<div style="line-height:2.4;padding:6px 0">' + "".join(pills) + "</div>",
                unsafe_allow_html=True,
            )

            with st.expander(f"Raw token IDs — {version}"):
                id_col, tok_col = st.columns(2)
                with id_col:
                    st.markdown("**IDs**")
                    st.code(str(ids))
                with tok_col:
                    st.markdown("**Tokens**")
                    st.code(str(tokens))

            st.markdown("&nbsp;")

    # ── 2. Fragmentation check ────────────────────────────────────────────────────
    st.subheader("2 · Fragmentation Check")
    st.caption(
        "For each probe term, shows whether the tokenizer learned it as a whole "
        "token (green) or fragmented it into subword pieces (red)."
    )

    if probe_terms:
        frag_results = {v: fragmentation_check(tok, probe_terms) for v, tok in tokenizers.items()}

        rows = []
        for i, term in enumerate(probe_terms):
            row = {"Term": term}
            for v in tokenizers:
                e = frag_results[v][i]
                row[v] = "● whole" if e["is_whole"] else f"{e['num_pieces']} pieces: " + " | ".join(e["tokens"])
            rows.append(row)

        frag_df = pl.DataFrame(rows).to_pandas()

        def _style_frag(val):
            if isinstance(val, str) and val.startswith("●"):
                return "background-color:#1b4332;color:#b7e4c7;font-weight:600;"
            if isinstance(val, str) and "pieces" in val:
                return "background-color:#3d0000;color:#ffb3b3;"
            return ""

        styled = frag_df.style.map(_style_frag, subset=list(tokenizers.keys()))
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Add probe terms in the settings expander to see fragmentation results.")

    # ── 3. Fertility & UNK rate ────────────────────────────────────────────────────
    st.subheader("3 · Fertility & UNK Rate")

    if not corpus_texts:
        st.info("No corpus data found for the selected dataset version to compute these metrics.")
    else:
        f_cols = st.columns(len(tokenizers))
        for col, (version, tok) in zip(f_cols, tokenizers.items()):
            f   = fertility(tok, corpus_texts)
            unk = unk_rate(tok, corpus_texts)
            with col:
                st.markdown(f"**{version}**")
                st.metric("Vocab Size", f"{tok.get_vocab_size():,}", 
                          help="Total number of tokens in the vocabulary")
                st.metric("Fertility",              f["fertility"],
                          help="tokens / word — lower = better compression")
                st.metric("Mean tokens / sentence", f["mean_tokens_per_sent"])
                st.metric("Std tokens / sentence",  f["std_tokens_per_sent"])
                st.metric("Min tokens / sentence",  f["min_tokens_per_sent"])
                st.metric("Max tokens / sentence",  f["max_tokens_per_sent"])
                st.markdown("&nbsp;")
                st.metric("UNK rate",  unk["unk_rate"],
                          help="Fraction of tokens that became [UNK] — lower = better coverage")
                st.metric("UNK count", unk["unk_count"])

    # ── 4. Vocabulary overlap (≥ 2 tokenizers) ────────────────────────────────────
    if len(tokenizers) >= 2:
        st.subheader("4 · Vocabulary Overlap")
        st.caption("Pairwise comparison of the token vocabularies between selected versions.")

        v_list = list(tokenizers.keys())
        for i in range(len(v_list)):
            for j in range(i + 1, len(v_list)):
                v1, v2 = v_list[i], v_list[j]
                ov     = vocab_overlap(tokenizers[v1], tokenizers[v2])

                st.markdown(f"**{v1}** vs **{v2}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Jaccard",         ov["jaccard"],
                          help="1.0 = identical vocabularies, 0.0 = completely different")
                c2.metric("Shared tokens",   ov["shared"])
                c3.metric(f"Only in {v1}",   ov["only_in_t1"])
                c4.metric(f"Only in {v2}",   ov["only_in_t2"])

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    with st.expander(f"Tokens only in {v1}"):
                        st.write(sorted(ov["only_in_t1_tokens"])[:100])
                with ec2:
                    with st.expander(f"Tokens only in {v2}"):
                        st.write(sorted(ov["only_in_t2_tokens"])[:100])
                with ec3:
                    with st.expander("Shared tokens (sample)"):
                        st.write(sorted(ov["shared_tokens"])[:100])

                st.markdown("&nbsp;")

    # ── 5. Token frequency from corpus ────────────────────────────────────────────
    if corpus_texts:
        st.subheader(f"5 · Top {top_n} Tokens by Corpus Frequency")
        st.caption(
            "How often each token actually appears across all descriptions. "
            "This reflects real usage in the domain."
        )

        freq_cols = st.columns(len(tokenizers))
        for col, (version, tok) in zip(freq_cols, tokenizers.items()):
            with col:
                st.markdown(f"**{version}**")
                freq = token_frequency(tok, corpus_texts, top_n=top_n)
                df   = pl.DataFrame({
                    "rank" : [f["rank"]  for f in freq],
                    "token": [f["token"] for f in freq],
                    "count": [f["count"] for f in freq],
                })
                st.dataframe(df.to_pandas(), use_container_width=True, hide_index=True)