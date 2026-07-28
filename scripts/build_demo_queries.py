"""Build the frontend demo-query catalog from real v2.0 training captions.

Run from the repository root while the retrieval API is available:

    uv run --project ml python scripts/build_demo_queries.py \
        --api-url http://127.0.0.1:8000/api/retrieval
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGEND_PATH = REPOSITORY_ROOT / "frontend/public/legend.json"
CAPTIONS_PATH = REPOSITORY_ROOT / "ml/data/text/patches_description.parquet"
OUTPUT_PATH = REPOSITORY_ROOT / "frontend/public/demo-queries.json"
SOURCE_VERSION = "v2.0"
PROMPT_STYLE = "llm_description"
QUERIES_PER_FEATURE = 3
MAX_QUERY_LENGTH = 500
DOMINANCE_TERMS = (
    "dominat",
    "majority",
    "overwhelming",
    "predomin",
    "primary",
    "expansive",
    "extensive",
    "largest",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        help=(
            "Optional retrieval endpoint used to keep only queries whose "
            "top-ranked patch contains the selected feature."
        ),
    )
    args = parser.parse_args()

    legend = json.loads(LEGEND_PATH.read_text(encoding="utf-8"))
    captions = (
        pl.read_parquet(CAPTIONS_PATH)
        .filter(
            (pl.col("source_version") == SOURCE_VERSION)
            & (pl.col("prompt_style") == PROMPT_STYLE)
            & (pl.col("text").str.len_chars() <= MAX_QUERY_LENGTH)
        )
        .select(["patch_id", "text"])
        .iter_rows(named=True)
    )
    caption_rows = list(captions)
    captions_by_patch = {int(row["patch_id"]): str(row["text"]) for row in caption_rows}

    feature_queries: dict[str, list[dict[str, int | str]]] = {}
    for code, metadata in legend.items():
        feature_name = str(metadata["long_description"])
        pattern = re.compile(rf"(?<![A-Za-z-]){re.escape(feature_name)}")
        candidates = [row for row in caption_rows if pattern.search(str(row["text"]))]
        candidates.sort(
            key=lambda row: _caption_score(str(row["text"]), feature_name),
            reverse=True,
        )
        if args.api_url:
            validated_candidates: list[dict[str, Any]] = []
            for row in candidates:
                if _top_result_contains_feature(
                    api_url=args.api_url,
                    query=str(row["text"]),
                    feature_name=feature_name,
                    captions_by_patch=captions_by_patch,
                ):
                    validated_candidates.append(row)
                if len(validated_candidates) == QUERIES_PER_FEATURE:
                    break
            candidates = validated_candidates
        feature_queries[code] = [
            {
                "patch_id": int(row["patch_id"]),
                "query": str(row["text"]),
            }
            for row in candidates[:QUERIES_PER_FEATURE]
        ]

    artifact = {
        "source_version": SOURCE_VERSION,
        "prompt_style": PROMPT_STYLE,
        "features": feature_queries,
    }
    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    unavailable = [code for code, queries in feature_queries.items() if not queries]
    print(
        f"Wrote {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)} with "
        f"{sum(map(len, feature_queries.values()))} verified v2 queries."
    )
    if unavailable:
        print(f"No matching v2 descriptions: {', '.join(unavailable)}")


def _caption_score(text: str, feature_name: str) -> tuple[int, int, float, int, int]:
    match = re.search(rf"(?<![A-Za-z-]){re.escape(feature_name)}", text)
    if match is None:
        return (0, 0, 0, 0, 0)

    position = match.start()
    context = text[max(0, position - 60) : position + len(feature_name) + 130].lower()
    percentages = [
        float(value)
        for value in re.findall(
            r"(\d+(?:\.\d+)?)%",
            text[max(0, position - 30) : position + len(feature_name) + 130],
        )
    ]
    return (
        int(position <= 60),
        sum(term in context for term in DOMINANCE_TERMS),
        max(percentages, default=0),
        -position,
        -len(text),
    )


def _top_result_contains_feature(
    *,
    api_url: str,
    query: str,
    feature_name: str,
    captions_by_patch: dict[int, str],
) -> bool:
    request = urllib.request.Request(
        api_url,
        data=json.dumps({"query": query, "top_k": 1}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload: Any = json.load(response)
    patch_id = int(payload["results"][0]["patch_id"])
    result_caption = captions_by_patch.get(patch_id, "")
    return bool(
        re.search(
            rf"(?<![A-Za-z-]){re.escape(feature_name)}",
            result_caption,
        )
    )


if __name__ == "__main__":
    main()
