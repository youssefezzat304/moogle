# LunarCLIP Vision Encoder

This folder exposes one CLIP-owned vision API:

```python
from lunar_clip.encoders.vision import LunarVisionEncoder
```

`LunarVisionEncoder` hides the model-specific loading and preprocessing needed
to turn pretrained vision checkpoints into one retrieval vector per patch.

## Supported Backends

`geo`

Loads the vision team's GEO retrieval encoder from:

```text
lunar_vision.model.geo.encoder.GeoEncoder
```

and supports:

```text
modality="geomap"
```

`wac`

Loads the vision team's WAC encoder from:

```text
lunar_vision.model.wac.encoder.WACEncoder
```

and supports:

```text
modality="wac"
```

`fusion`

Loads the vision team's fusion encoder from:

```text
lunar_vision.model.fusion.encoder.FusionVisionEncoder
```

and supports:

```text
modality="wac"
```

All three backends implement `VisionEncoderBackend.encode_retrieval`. Freezing
is controlled by `LunarVisionEncoder`: `freeze_encoder=True` freezes pretrained
weights and leaves retrieval tokens trainable, while `False` leaves the full
backend trainable.

## Usage

```python
encoder = LunarVisionEncoder(
    encoder="geo",
    checkpoint_path="artifacts/vision_models/geo2geo/best.pt",
    freeze_encoder=True,
)
```

The encoder returns a `RetrievalBatch` from:

```python
encoder.encode_image(batch, modality="geomap")
```

with vectors shaped:

```text
(batch_size, output_dim)
```

For Geo, the batch should include the RGB geomap patch under `"original"` when
available. LunarCLIP's training collate path preserves that tensor.

For WAC, the batch should include the normalized WAC patch under `"tensor"`.

Fusion also consumes the WAC `"tensor"` and requires its native 512 x 512 input.
GEO and WAC can resize their positional embeddings for compatible patch grids.
