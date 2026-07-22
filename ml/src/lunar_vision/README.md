# Lunar Vision

This package is reserved for the vision team's native model code. It mirrors
the role of `lunar_text`: model definitions, training code, preprocessing,
checkpoints, and vision-specific utilities live here.

The CLIP package owns the retrieval-facing bridge through:

```python
from lunar_clip.encoders.vision import LunarVisionEncoder
```

## Ownership Boundary

Use `lunar_vision` for implementation details that belong to the vision team:

- GEO, WAC, and Fusion architectures, training code, and checkpoint loaders.
- Vision-only datasets, transforms, losses, metrics, and utilities.

Use `lunar_clip.encoders.vision.LunarVisionEncoder` for CLIP alignment:

- loading pretrained vision checkpoints for CLIP use
- converting model-specific outputs into one retrieval vector per patch
- exposing `encode_image(batch, modality)` to `LunarCLIPModel`

LunarCLIP should not need to know whether a vision model is a transformer,
autoencoder, CNN, or another architecture.

## Current CLIP-Facing API

The active CLIP path uses one centralized vision encoder:

```python
vision_encoder = LunarVisionEncoder(
    encoder="geo",
    checkpoint_path="artifacts/vision_models/geo2geo/best.pt",
    freeze_encoder=True,
)
```

Supported backend names:

- `geo`: implemented through `lunar_vision.model.geo.encoder.GeoEncoder`
- `wac`: implemented through `lunar_vision.model.wac.encoder.WACEncoder`
- `fusion`: implemented through
  `lunar_vision.model.fusion.encoder.FusionVisionEncoder`

Each native encoder implements the abstract
`lunar_vision.model.clip_backend.VisionEncoderBackend` contract and returns one
retrieval vector per input image.

The returned retrieval batch contains vectors shaped:

```text
(batch_size, output_dim)
```

## How LunarCLIP Uses Vision

The high-level training flow is:

```text
caption text -> LunarTextEncoder.encode_text(...) -> text retrieval vector
image patch  -> LunarVisionEncoder.encode_image(...) -> image retrieval vector
text vector  -> CLIP text projection
image vector -> CLIP image projection
projected vectors -> contrastive loss
```

## Training Example

The config-driven CLIP command is:

```bash
uv run python main.py clip train --config configs/clip/bpe_geo.yaml
```

The relevant config shape is:

```yaml
text:
  encoder: bpe
  tokenizer_path: artifacts/tokenizers/bpe/v4.0/tokenizer.json
  checkpoint_path: artifacts/text_models/bpe/step_085000.ckpt
  freeze_encoder: true

vision:
  encoder: geo
  checkpoint_path: artifacts/vision_models/geo2geo/best.pt
  freeze_encoder: true
```

With `freeze_encoder: true`, LunarVisionEncoder freezes pretrained backend
weights and leaves retrieval tokens trainable. With `freeze_encoder: false`,
all backend parameters remain trainable. GEO and WAC support positional-grid
resizing; Fusion intentionally requires its native 512 x 512 input.
