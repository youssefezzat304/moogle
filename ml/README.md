# LunarCLIP

LunarCLIP aligns lunar patch descriptions and vision patches in one shared
embedding space for multimodal retrieval. The text and vision backends remain
independent; LunarCLIP owns the projection, normalization, contrastive loss,
and retrieval-facing APIs.

## Model Flow

```text
caption text
    -> LunarTextEncoder.encode_text(...)
    -> text retrieval vector
    -> CLIP text projection
    -> L2 normalization

image patch
    -> LunarVisionEncoder.encode_image(..., modality=...)
    -> image retrieval vector
    -> CLIP image projection
    -> L2 normalization

normalized text vectors @ normalized image vectors.T
    -> scaled similarity logits
    -> symmetric multi-positive contrastive loss during training
```

Patch IDs define positive text-image pairs, so a batch may contain more than
one caption for an image. When IDs are not supplied through the Python API,
the model falls back to the standard one-to-one symmetric CLIP loss.

Each vision sample produces one vector per patch. The supported vision
backends and their dataset modalities are:

| Backend | Modality | Status |
| --- | --- | --- |
| `geo` | `geomap` | Implemented |
| `wac` | `wac` | Implemented |
| `fusion` | `wac` | Implemented |

The modality argument prevents a backend from accidentally receiving the wrong
vision tensor. Checkpoints are loaded by `LunarVisionEncoder`, while patch
encoding, positional embeddings, and retrieval-token pooling stay in
`lunar_vision`. GEO and WAC can resize positional embeddings for compatible
input sizes; Fusion expects its native 512 x 512 input.

The supported text backends are `bpe`, `wordpiece`, and `ngram`. All three use
the same retrieval sequence:

```text
[RETRIEVAL] [SOS] caption tokens [EOS] [PAD] ...
```

The hidden state for `[RETRIEVAL]` becomes the text retrieval vector. Each
tokenizer must provide `[RETRIEVAL]`, `[SOS]`, `[EOS]`, `[PAD]`, `[MASK]`, and
`[UNK]`. The repository's BPE, WordPiece, and N-gram checkpoints are covered by
smoke tests that load the real artifacts and run a backward pass.

## Setup

Install the locked project dependencies with:

```bash
uv sync
```

Inspect the available commands with:

```bash
uv run python main.py --help
uv run python main.py clip --help
```

Before training, verify that the tokenizer, text checkpoint, vision checkpoint,
caption parquet, and machine-specific `vision_root` in the selected YAML config
exist. The checked-in GEO config uses
`artifacts/vision_models/geo2geo/best.pt`.

## Train LunarCLIP

Training is configured with YAML files under `configs/clip/`:

```bash
uv run python main.py clip train --config configs/clip/bpe_geo.yaml
uv run python main.py clip train --config configs/clip/bpe_wac.yaml
uv run python main.py clip train --config configs/clip/bpe_fusion.yaml
```

These are the currently populated training configs. WordPiece and N-gram are
available through `LunarTextEncoder`, but need a completed CLIP YAML before a
config-driven run.

The config selects:

- the text encoder, tokenizer, and text checkpoint;
- the vision encoder and vision checkpoint;
- the dataset paths, patch size, stride, caption policy, and evaluation
  fraction;
- projection size, temperature, batch size, epochs, optimizer, optional text
  encoder learning rate, device, and output directory.

`caption_policy` supports `first`, `sample_one`, and
`two_llm_descriptions`. The last policy selects the v1.0 and v2.0
`llm_description` captions for each patch and trains them as two positives for
the same image.

Set `data.eval_fraction` to a value greater than zero to create a validation
split. The training CLI evaluates that split after training and logs full-index
metrics under `/eval`. A test dataset can also be supplied through the Python
training API.

Training writes checkpoints under the configured output directory:

```text
<output_dir>/checkpoints/best.ckpt
<output_dir>/evaluations/best-eval.json
```

When a test dataset is supplied, `best-test.json` is written alongside the
evaluation artifact. The best checkpoint is selected by evaluation loss when a
validation split exists, otherwise by training loss.

TensorBoard logs are written under the configured output directory and run
name. Start TensorBoard with:

```bash
uv run tensorboard --logdir results/clip
```

Training-time retrieval metrics are in-batch diagnostics:

```text
in_batch_text_to_image_top1
in_batch_image_to_text_top1
in_batch_retrieval_top1
```

After training, full-index metrics compare every query with the complete
evaluation index. They use names such as:

```text
full_text_to_image_recall@1/eval
full_image_to_text_recall@10/eval
full_text_to_image_mean_rank/eval
full_mean_recall/eval
```

Multi-caption evaluation also reports version-specific text-to-image metrics,
for example `full_v1_0_text_to_image_recall@1/eval`.

Use the `full_*` metrics for final model comparison, not the in-batch metrics.

## Encode Text From The CLI

Encode one caption with the supported text CLI command:

```bash
uv run python main.py clip encode-text \
  --encoder bpe \
  --checkpoint artifacts/text_models/bpe/step_085000.ckpt \
  --tokenizer artifacts/tokenizers/bpe/v4.0/tokenizer.json \
  --text "A cratered lunar plain with a sharp terrain boundary."
```

The command prints the resulting text retrieval vector. This CLI command is
currently limited to BPE; WordPiece and N-gram are supported by the Python
adapter and training path.

The `clip encode-image` command is listed in the CLI help, but loading a patch
by ID is not wired yet. Use the `LunarVisionEncoder` Python API for image
encoding.

## Useful Locations

```text
main.py                                  CLI entry point
configs/clip/                            CLIP training configurations
docs/current_clip_architecture.md        Implemented architecture and limitations
docs/target_clip_architecture.md         Ownership boundaries and remaining goals
src/lunar_clip/model/                    CLIP model and projections
src/lunar_clip/encoders/                 Text and vision adapters
src/lunar_clip/training/                 Lightning training loop and metrics
src/lunar_clip/retrieval/                Full-index metrics and vector store
src/lunar_vision/                        Native GEO, WAC, and Fusion implementations
artifacts/text_models/                   Text checkpoints
artifacts/vision_models/                 Vision checkpoints
results/                                 Training runs and logs
```
