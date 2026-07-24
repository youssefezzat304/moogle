from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import yaml
from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.model.lunar_clip_model import LunarCLIPModel
from lunar_text.model.bpe.config import ModelConfig
from lunar_vision.model.geo.encoder import GeoEncoder

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROMOTED_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PromotedEncoderArchitecture:
    """A named encoder and its architecture-specific manifest parameters."""

    encoder: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class PromotedModelManifest:
    model_id: str
    modality: str
    projection_dim: int
    temperature: float
    preprocessing_id: str
    text: PromotedEncoderArchitecture
    vision: PromotedEncoderArchitecture
    checkpoint_path: str
    checkpoint_sha256: str
    tokenizer_path: str
    tokenizer_sha256: str


@dataclass(frozen=True)
class LoadedPromotedModel:
    model: LunarCLIPModel
    manifest: PromotedModelManifest
    manifest_path: Path


TextEncoderFactory = Callable[
    [PromotedEncoderArchitecture, Path],
    LunarTextEncoder,
]
VisionEncoderFactory = Callable[
    [PromotedEncoderArchitecture],
    LunarVisionEncoder,
]


@dataclass(frozen=True)
class VisionEncoderRegistration:
    factory: VisionEncoderFactory
    modality: str


def load_promoted_lunar_clip_model(
    manifest_path: str | Path,
) -> LoadedPromotedModel:
    """Reconstruct a promoted LunarCLIP model and strictly load all weights."""

    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = _read_manifest(resolved_manifest_path)

    checkpoint_path = _resolve_storage_file(
        resolved_manifest_path,
        manifest.checkpoint_path,
    )
    tokenizer_path = _resolve_storage_file(
        resolved_manifest_path,
        manifest.tokenizer_path,
    )
    _verify_file_digest(
        checkpoint_path,
        expected=manifest.checkpoint_sha256,
        label="checkpoint",
    )
    _verify_file_digest(
        tokenizer_path,
        expected=manifest.tokenizer_sha256,
        label="tokenizer",
    )

    text_adapter = _build_registered_text_encoder(
        manifest.text,
        tokenizer_path,
    )
    vision_adapter = _build_registered_vision_encoder(
        manifest.vision,
        modality=manifest.modality,
    )
    model = LunarCLIPModel(
        text_adapter=text_adapter,
        vision_adapter=vision_adapter,
        projection_dim=manifest.projection_dim,
        temperature=manifest.temperature,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Promoted checkpoint must contain a mapping.")
    _verify_checkpoint_modality(checkpoint, expected=manifest.modality)
    state_dict = _model_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return LoadedPromotedModel(
        model=model,
        manifest=manifest,
        manifest_path=resolved_manifest_path,
    )


def _read_manifest(path: Path) -> PromotedModelManifest:
    if not path.is_file():
        raise FileNotFoundError(f"Promoted model manifest not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("Promoted model manifest must contain a mapping.")
    if raw.get("schema_version") != PROMOTED_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported promoted model manifest schema_version.")

    model = _mapping(raw, "model")
    architecture = _mapping(raw, "architecture")
    text = _mapping(architecture, "text")
    vision = _mapping(architecture, "vision")
    files = _mapping(raw, "files")
    checkpoint = _mapping(files, "checkpoint")
    tokenizer = _mapping(files, "tokenizer")
    try:
        return PromotedModelManifest(
            model_id=_non_empty_string(raw, "model_id"),
            modality=_non_empty_string(model, "modality"),
            projection_dim=_positive_int(model, "projection_dim"),
            temperature=_positive_float(model, "temperature"),
            preprocessing_id=_non_empty_string(model, "preprocessing_id"),
            text=_encoder_architecture(text),
            vision=_encoder_architecture(vision),
            checkpoint_path=_non_empty_string(checkpoint, "path"),
            checkpoint_sha256=_sha256(checkpoint, "sha256"),
            tokenizer_path=_non_empty_string(tokenizer, "path"),
            tokenizer_sha256=_sha256(tokenizer, "sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid promoted model manifest.") from exc


def _build_registered_text_encoder(
    architecture: PromotedEncoderArchitecture,
    tokenizer_path: Path,
) -> LunarTextEncoder:
    factory = TEXT_ENCODER_FACTORIES.get(architecture.encoder)
    if factory is None:
        raise ValueError(
            f"Unsupported promoted text encoder {architecture.encoder!r}. "
            f"Registered encoders: {sorted(TEXT_ENCODER_FACTORIES)}."
        )
    return factory(architecture, tokenizer_path)


def _build_registered_vision_encoder(
    architecture: PromotedEncoderArchitecture,
    *,
    modality: str,
) -> LunarVisionEncoder:
    registration = VISION_ENCODER_FACTORIES.get(architecture.encoder)
    if registration is None:
        raise ValueError(
            f"Unsupported promoted vision encoder {architecture.encoder!r}. "
            f"Registered encoders: {sorted(VISION_ENCODER_FACTORIES)}."
        )
    if modality != registration.modality:
        raise ValueError(
            f"Vision encoder {architecture.encoder!r} requires modality "
            f"{registration.modality!r}; got {modality!r}."
        )
    return registration.factory(architecture)


def _build_bpe_text_encoder(
    architecture: PromotedEncoderArchitecture,
    tokenizer_path: Path,
) -> LunarTextEncoder:
    parameters = architecture.parameters
    embed_dim = _positive_int(parameters, "embed_dim")
    num_heads = _positive_int(parameters, "num_heads")
    if embed_dim % num_heads:
        raise ValueError("BPE embed_dim must be divisible by num_heads.")

    model_config = ModelConfig(
        vocab_size=_positive_int(parameters, "vocab_size"),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=_positive_int(parameters, "num_layers"),
        ffn_dim=_positive_int(parameters, "ffn_dim"),
        dropout=_non_negative_float(parameters, "dropout"),
        max_seq_len=_positive_int(parameters, "max_seq_len"),
        pad_token_id=_non_negative_int(parameters, "pad_token_id"),
        layer_norm_eps=_positive_float(parameters, "layer_norm_eps"),
    )
    return LunarTextEncoder(
        tokenizer_path=str(tokenizer_path),
        encoder=architecture.encoder,
        checkpoint_path=None,
        model_config=model_config,
        max_length=model_config.max_seq_len,
        freeze_encoder=False,
    )


def _build_geo_vision_encoder(
    architecture: PromotedEncoderArchitecture,
) -> LunarVisionEncoder:
    parameters = architecture.parameters
    hidden_dim = _positive_int(parameters, "hidden_dim")
    num_heads = _positive_int(parameters, "num_heads")
    image_channels = _positive_int(parameters, "image_channels")
    if hidden_dim % num_heads:
        raise ValueError("Geo hidden_dim must be divisible by num_heads.")
    if image_channels != 3:
        raise ValueError("The Geo encoder must consume three RGB channels.")

    backend = GeoEncoder(
        patch_size=_positive_int(parameters, "token_patch_size"),
        image_size=_positive_int(parameters, "pretraining_image_size"),
        hidden_dim=hidden_dim,
        nheads=num_heads,
        num_layers=_positive_int(parameters, "num_layers"),
        img_channels=image_channels,
    )
    return LunarVisionEncoder(
        encoder=architecture.encoder,
        backend=backend,
        freeze_encoder=False,
    )


TEXT_ENCODER_FACTORIES: Mapping[str, TextEncoderFactory] = MappingProxyType(
    {
        "bpe": _build_bpe_text_encoder,
    }
)
VISION_ENCODER_FACTORIES: Mapping[str, VisionEncoderRegistration] = (
    MappingProxyType(
        {
            "geo": VisionEncoderRegistration(
                factory=_build_geo_vision_encoder,
                modality="geomap",
            ),
        }
    )
)


def _encoder_architecture(
    value: dict[str, Any],
) -> PromotedEncoderArchitecture:
    encoder = _non_empty_string(value, "encoder")
    return PromotedEncoderArchitecture(
        encoder=encoder,
        parameters=MappingProxyType(
            {
                key: item
                for key, item in value.items()
                if key != "encoder"
            }
        ),
    )


def _resolve_storage_file(manifest_path: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("Promoted model file paths must be relative.")
    try:
        storage_root = manifest_path.parents[2]
    except IndexError as exc:
        raise ValueError("Manifest must be stored below a storage root.") from exc
    resolved = (manifest_path.parent / relative_path).resolve()
    if not resolved.is_relative_to(storage_root):
        raise ValueError("Promoted model file path escapes the storage root.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Promoted model file not found: {resolved}")
    return resolved


def _verify_file_digest(path: Path, *, expected: str, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Promoted {label} SHA-256 mismatch: expected {expected}, got {actual}."
        )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint_modality(
    checkpoint: dict[str, Any],
    *,
    expected: str,
) -> None:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise TypeError("Checkpoint is missing hyper_parameters.")
    training_config = hyper_parameters.get("training_config")
    if not isinstance(training_config, dict):
        raise TypeError("Checkpoint is missing training_config metadata.")
    if training_config.get("modality") != expected:
        raise ValueError("Checkpoint modality does not match the promoted manifest.")


def _model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("Checkpoint is missing its state_dict.")
    if any(
        not isinstance(key, str) or not key.startswith("model.") for key in raw_state
    ):
        raise ValueError("Checkpoint state_dict must use the Lightning model prefix.")
    state_dict = {key.removeprefix("model."): value for key, value in raw_state.items()}
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("Checkpoint state_dict values must be tensors.")
    return state_dict


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TypeError(f"Manifest field '{key}' must be a mapping.")
    return item


def _non_empty_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Manifest field '{key}' must be a non-empty string.")
    return item


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ValueError(f"Manifest field '{key}' must be a positive integer.")
    return item


def _non_negative_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ValueError(f"Manifest field '{key}' must be a non-negative integer.")
    return item


def _positive_float(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
        raise ValueError(f"Manifest field '{key}' must be positive.")
    return float(item)


def _non_negative_float(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
        raise ValueError(f"Manifest field '{key}' must be non-negative.")
    return float(item)


def _sha256(value: dict[str, Any], key: str) -> str:
    digest = _non_empty_string(value, key)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"Manifest field '{key}' must be a lowercase SHA-256.")
    return digest
