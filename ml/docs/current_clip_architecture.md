# Current CLIP Architecture (last update: 21.07.2026)

This document describes the implemented CLIP architecture and the remaining
integration limitations.

## Goal

LunarCLIP connects one text encoder and one vision encoder into a shared
retrieval space.

```text
caption text -> text encoder -> text vector
image patch  -> vision encoder -> image vector

text vector  -> text projection  -> normalized text embedding
image vector -> image projection -> normalized image embedding

normalized embeddings -> contrastive logits/loss
```

`lunar_clip` owns the retrieval task. It should decide which text and vision
backends are used, project their output vectors into a shared dimension, and
compute contrastive loss.

## Text Side

The current CLIP text entry point is:

```text
src/lunar_clip/encoders/text/lunar_text_encoder.py
```

`LunarTextEncoder` loads one backend by name:

```text
bpe
wordpiece
ngram
```

Each backend is expected to return hidden states shaped:

```text
(batch_size, sequence_length, hidden_dim)
```

The shared text pooling rule is retrieval-token pooling:

```text
hidden_states[:, 0, :]
```

For that to work, encoded CLIP text must start with:

```text
[RETRIEVAL]
```

The expected special-token contract is:

```text
[RETRIEVAL], [SOS], [EOS], [PAD], [MASK], [UNK]
```

All three backends implement the shared
`lunar_text.model.clip_backend.TextEncoderBackend` retrieval contract. The
checked-in BPE, WordPiece, and N-gram tokenizer/checkpoint pairs are covered by
smoke tests that load the real artifacts, encode text, and run a backward pass.
WordPiece restores `type_vocab_size=2` from its training configuration or
checkpoint when needed.

## Vision Side

The current CLIP vision entry point is:

```text
src/lunar_clip/encoders/vision/lunar_vision_encoder.py
```

`LunarVisionEncoder` loads one backend by name:

```text
geo
wac
fusion
```

Each native backend subclasses the abstract
`lunar_vision.model.clip_backend.VisionEncoderBackend` contract and implements:

```python
encode_retrieval(image_tensor) -> Tensor  # shape (B, D)
```

The vision path uses retrieval-token pooling:

```text
image
-> patch tokens
-> resized positional embeddings
-> retrieval token prepended
-> transformer
-> token 0 as image retrieval vector
```

The backend owns its model details, including:

```text
conv_proj
pos_embed
retrieval
transformer
```

GEO and WAC resize their absolute positional embeddings when the input patch
grid changes. Fusion uses its checkpoint's fixed 512 x 512 WAC-side grid and
rejects incompatible resolutions. `LunarVisionEncoder` validates the modality,
selects the input tensor, and delegates retrieval encoding to the backend.

For all three vision backends, `freeze_encoder=True` freezes pretrained weights
while leaving the retrieval token trainable. `freeze_encoder=False` leaves the
whole backend trainable.

## CLIP Model

The central model is:

```text
src/lunar_clip/model/lunar_clip_model.py
```

It receives:

```text
text_batch
image_batch
modality
text_patch_ids
image_patch_ids
```

Then it:

1. Encodes text through `LunarTextEncoder`.
2. Encodes image through `LunarVisionEncoder`.
3. Applies `text_projection`.
4. Applies `image_projection`.
5. Normalizes both embeddings.
6. Computes the full text-by-image similarity matrix.
7. Computes symmetric multi-positive contrastive loss using matching patch IDs.

The CLIP projection heads are separate from the native encoder hidden
dimensions. They are the trainable bridge into the shared retrieval space.

## Training

The training entry point is:

```text
src/lunar_clip/training/train_clip.py
```

The training batch contains:

```text
text: list[str]
vision: dict[modality, dict[str, Tensor]]
text_patch_id: Tensor
image_patch_id: Tensor
patch_id: Tensor  # compatibility alias for image_patch_id
coords: Tensor
caption_metadata: list[dict[str, str]]
text_version: list[str]
```

The number of text rows may exceed the number of image rows. The
`two_llm_descriptions` policy, for example, emits the v1.0 and v2.0
`llm_description` captions as two positive texts for each image. The loss and
metrics use patch IDs instead of diagonal positions to identify positives.

Training logs use explicitly named in-batch retrieval metrics:

```text
in_batch_text_to_image_top1
in_batch_image_to_text_top1
in_batch_retrieval_top1
```

They compare each text only against images in the same batch. They are useful
diagnostics, but they are not full-dataset retrieval evaluation. Full-index
evaluation logs `full_*` metrics after training by comparing validation/test
queries against the full evaluation embedding index. Multi-caption evaluation
also exports per-source-version text-to-image metrics.

The optimizer can assign `text_encoder_learning_rate` to a trainable text
backend while using `learning_rate` for projections and other trainable
parameters. Training retains only `<output_dir>/checkpoints/best.ckpt`, reloads
it before final evaluation, and writes structured results under
`<output_dir>/evaluations/`.

## Remaining Limitations

- `clip encode-text` currently exposes only the BPE backend, even though the
  Python adapter and training path support BPE, WordPiece, and N-gram.
- `clip encode-image` is visible in CLI help, but patch lookup and image loading
  are not wired yet.
- Only the BPE GEO, WAC, and Fusion CLIP YAML files are populated. WordPiece and
  N-gram still need complete run configs for config-driven training.
- A `.ckpt` stores the training configuration, but does not by itself store all
  tokenizer and pretrained backend paths needed to reconstruct the adapters.
  The exported evaluation JSON contains the fuller model, dataset, and training
  metadata.
- The vision freeze policy still recognizes trainable retrieval tokens by
  parameter name. A backend-owned freeze contract would be less coupled.
- Training/evaluation still extracts modality-specific tensor keys outside the
  vision backend. Moving that preparation fully behind the adapter would make
  the batch contract cleaner.
- The best checkpoint is selected by loss, not a full-index retrieval metric;
  full-index evaluation currently runs after training.
