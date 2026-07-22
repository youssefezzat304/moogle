from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.model.lunar_clip_model import LunarCLIPModel
from lunar_text.model.bpe.config import ModelConfig
from lunar_vision.model.geo.encoder import GeoEncoder
from lunar_vision.model.geo.model import Encoder

BPE_TOKENIZER_PATH = Path("artifacts/tokenizers/bpe/v4.0/tokenizer.json")


# Small architecture for fast, deterministic tests. Mirrors the real Geo2Geo
# training config's field names (enc_patch=ViT token size, patch_size=tile
# size) but at a fraction of the size.
TEST_ARGS = {"enc_patch": 4, "hidden_dim": 8, "nheads": 2, "num_layers": 1, "patch_size": 16}


def _fabricate_checkpoint(
    tmp_path: Path,
    *,
    drop_retrieval: bool,
    extra_missing_key: bool = False,
    args: dict = TEST_ARGS,
) -> Path:
    """Build a Geo2Geo-shaped checkpoint like train.py would save, optionally
    without a 'retrieval' key to simulate a pre-retrieval-token checkpoint."""
    encoder = Encoder(
        patch_size=args["enc_patch"],
        image_size=args["patch_size"],
        img_channels=3,
        hidden_dim=args["hidden_dim"],
        nheads=args["nheads"],
        num_layers=args["num_layers"],
    )
    state_dict = {f"encoder.{key}": value for key, value in encoder.state_dict().items()}
    if drop_retrieval:
        del state_dict["encoder.retrieval"]
    if extra_missing_key:
        del state_dict["encoder.conv_proj.weight"]

    checkpoint_path = tmp_path / "fabricated.pt"
    torch.save({"model": state_dict, "args": args}, checkpoint_path)
    return checkpoint_path


def test_load_from_checkpoint_tolerates_missing_retrieval_key(tmp_path: Path) -> None:
    checkpoint_path = _fabricate_checkpoint(tmp_path, drop_retrieval=True)

    geo_encoder = GeoEncoder.load_from_checkpoint(str(checkpoint_path))

    assert isinstance(geo_encoder.encoder.retrieval, torch.nn.Parameter)
    assert geo_encoder.output_dim == TEST_ARGS["hidden_dim"]


def test_load_from_checkpoint_raises_on_other_missing_keys(tmp_path: Path) -> None:
    checkpoint_path = _fabricate_checkpoint(tmp_path, drop_retrieval=True, extra_missing_key=True)

    with pytest.raises(RuntimeError, match="Missing keys"):
        GeoEncoder.load_from_checkpoint(str(checkpoint_path))


def test_encode_retrieval_at_native_resolution(tmp_path: Path) -> None:
    checkpoint_path = _fabricate_checkpoint(tmp_path, drop_retrieval=True)
    geo_encoder = GeoEncoder.load_from_checkpoint(str(checkpoint_path))

    native_size = TEST_ARGS["patch_size"]  # 16 -> 4x4 grid at enc_patch=4, matches training grid exactly
    batch = torch.randn(2, 3, native_size, native_size)

    vectors = geo_encoder.encode_retrieval(batch)

    assert vectors.shape == (2, TEST_ARGS["hidden_dim"])


def test_encode_retrieval_resizes_pos_embed_for_larger_clip_resolution(tmp_path: Path) -> None:
    """The real-world case: Geo2Geo trains at image_size=256, enc_patch=16
    (16x16=256 pos_embed grid), but CLIP feeds 512x512 patches with the same
    enc_patch=16, giving a 32x32=1024 grid. This must trigger a bicubic
    pos_embed resize with no tensor shape mismatch, not a crash."""
    checkpoint_path = _fabricate_checkpoint(tmp_path, drop_retrieval=True)
    geo_encoder = GeoEncoder.load_from_checkpoint(str(checkpoint_path))

    native_size = TEST_ARGS["patch_size"]
    clip_size = native_size * 2  # forces the pos_embed grid to double each side
    batch = torch.randn(3, 3, clip_size, clip_size)

    vectors = geo_encoder.encode_retrieval(batch)

    assert vectors.shape == (3, TEST_ARGS["hidden_dim"])


def test_lunar_vision_encoder_end_to_end_for_geo(tmp_path: Path) -> None:
    checkpoint_path = _fabricate_checkpoint(tmp_path, drop_retrieval=True)

    vision_encoder = LunarVisionEncoder(
        encoder="geo",
        checkpoint_path=str(checkpoint_path),
        freeze_encoder=True,
    )

    native_size = TEST_ARGS["patch_size"]
    batch = torch.randint(0, 256, (2, 3, native_size, native_size), dtype=torch.uint8)

    retrieval_batch = vision_encoder.encode_image(batch, modality="geomap")

    assert retrieval_batch.vectors.shape == (2, TEST_ARGS["hidden_dim"])

    trainable = {
        name for name, parameter in vision_encoder.named_parameters() if parameter.requires_grad
    }
    assert trainable == {"model.encoder.retrieval"}, (
        "freeze_encoder=True must leave only the retrieval token trainable, "
        f"got {trainable}"
    )


@pytest.mark.skipif(
    not BPE_TOKENIZER_PATH.exists(),
    reason=f"{BPE_TOKENIZER_PATH} not found.",
)
def test_lunar_clip_model_integrates_geo_vision_with_bpe_text(tmp_path: Path) -> None:
    """End-to-end proof that the geo retrieval token is wired into the full
    CLIP contrastive path, not just reachable in isolation: a real
    LunarCLIPModel forward+backward pass, at CLIP's actual 512x512 patch
    size, must produce paired logits and a gradient on the retrieval token.

    Uses enc_patch=16 (the real ViT token size, unlike the tiny enc_patch=4
    in TEST_ARGS used elsewhere in this file) so the 512x512 CLIP image
    resizes to the real ~32x32=1024-token grid instead of an unrealistic
    128x128=16384-token grid that would make attention prohibitively slow.
    """
    realistic_args = {**TEST_ARGS, "enc_patch": 16, "patch_size": 64}
    checkpoint_path = _fabricate_checkpoint(tmp_path, drop_retrieval=True, args=realistic_args)
    vision_adapter = LunarVisionEncoder(
        encoder="geo", checkpoint_path=str(checkpoint_path), freeze_encoder=True,
    )

    text_config = ModelConfig(embed_dim=16, num_heads=2, num_layers=1, ffn_dim=32, max_seq_len=16)
    text_adapter = LunarTextEncoder(
        encoder="bpe",
        tokenizer_path=str(BPE_TOKENIZER_PATH),
        checkpoint_path=None,
        model_config=text_config,
        freeze_encoder=True,
    )

    model = LunarCLIPModel(
        text_adapter=text_adapter, vision_adapter=vision_adapter, projection_dim=8, temperature=0.07,
    )

    texts = ["A cratered lunar plain.", "A smoother highland patch."]
    image_batch = {
        "original": torch.randint(0, 256, (2, 3, REAL_CLIP_PATCH_SIZE, REAL_CLIP_PATCH_SIZE), dtype=torch.uint8),
    }

    output = model(text_batch=texts, image_batch=image_batch, modality="geomap", return_loss=True)

    assert output.text_embeds.shape == (2, 8)
    assert output.image_embeds.shape == (2, 8)
    assert output.logits_per_text.shape == (2, 2)

    output.loss.backward()
    retrieval_grad = model.vision_adapter.model.encoder.retrieval.grad
    assert retrieval_grad is not None and retrieval_grad.abs().sum().item() > 0, (
        "Geo retrieval token must receive gradient from the CLIP contrastive loss."
    )


REAL_CHECKPOINT_PATH = Path("artifacts/vision_models/geo2geo/best.pt")
REAL_CLIP_PATCH_SIZE = 512  # matches data.patch_size in configs/clip/bpe_geo.yaml


def _real_checkpoint_is_loadable() -> bool:
    if not REAL_CHECKPOINT_PATH.exists():
        return False
    try:
        torch.load(REAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _real_checkpoint_is_loadable(),
    reason="artifacts/vision_models/geo2geo/best.pt is absent or is a Git LFS pointer stub; run `git lfs pull` to enable this check.",
)
def test_real_geo2geo_checkpoint_handles_clip_patch_size() -> None:
    geo_encoder = GeoEncoder.load_from_checkpoint(str(REAL_CHECKPOINT_PATH))
    batch = torch.randn(2, 3, REAL_CLIP_PATCH_SIZE, REAL_CLIP_PATCH_SIZE)

    vectors = geo_encoder.encode_retrieval(batch)

    assert vectors.shape == (2, geo_encoder.output_dim)
