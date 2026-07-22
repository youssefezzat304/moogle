from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EVALUATION_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalEvaluationMetadata:
    model: dict[str, Any]
    dataset: dict[str, Any]
    training: dict[str, Any]


@dataclass(frozen=True)
class RetrievalEvaluationArtifact:
    checkpoint: dict[str, str]
    evaluation: dict[str, Any]
    metrics: dict[str, float]
    model: dict[str, Any]
    dataset: dict[str, Any]
    training: dict[str, Any]
    schema_version: int = EVALUATION_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_retrieval_evaluation_artifact(
    artifact: RetrievalEvaluationArtifact,
    output_dir: str | Path,
) -> Path:
    split = artifact.evaluation.get("split")
    if not isinstance(split, str) or not split:
        raise ValueError("Retrieval evaluation artifacts require a non-empty split name.")

    checkpoint_name = artifact.checkpoint.get("name")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError("Retrieval evaluation artifacts require a checkpoint name.")

    artifact_dir = Path(output_dir) / "evaluations"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{Path(checkpoint_name).stem}-{split}.json"
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return artifact_path
