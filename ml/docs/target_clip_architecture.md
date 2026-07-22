# Target CLIP Architecture

This document records the intended ownership boundaries and the work that
remains. The shared text/vision retrieval contracts, backend-owned vision
encoding, and full-index evaluation described below are now implemented.

## Goal

The final architecture should keep responsibilities simple:

- `lunar_text` owns text tokenization, text checkpoints, and text retrieval
  features.
- `lunar_vision` owns vision model internals, vision checkpoints, positional
  embeddings, retrieval tokens, and vision retrieval features.
- `lunar_clip` owns CLIP projection, normalization, contrastive loss, retrieval
  metrics, and training/evaluation orchestration.

In the target design, CLIP should not know how a BPE model tokenizes text or how
a GEO/WAC model resizes positional embeddings. CLIP should only ask each backend
for one retrieval vector.

## Target Flow

```text
caption text -> text_backend.encode_retrieval(...)   -> text vector
image patch  -> vision_backend.encode_retrieval(...) -> image vector

text vector  -> text_projection  -> normalized text embedding
image vector -> image_projection -> normalized image embedding

normalized embeddings -> contrastive loss / retrieval evaluation
```

Both retrieval vectors should be shaped:

```text
(batch_size, hidden_dim)
```

## Text Backend Contract

Every CLIP-compatible text backend should support the same retrieval contract.

Required special tokens:

```text
[RETRIEVAL], [SOS], [EOS], [PAD], [MASK], [UNK]
```

Expected CLIP input layout:

```text
[RETRIEVAL], [SOS], encoded tokens..., [EOS], [PAD]...
```

Expected retrieval vector:

```text
hidden_states[:, 0, :]
```

The preferred backend-facing API is:

```python
encode_retrieval(texts_or_token_batch) -> Tensor  # shape (B, D)
```

`LunarTextEncoder` is the CLIP-side adapter, and backend behavior is consistent
across:

```text
bpe
wordpiece
ngram
```

The N-gram backend is a neural Transformer backend and uses token IDs and the
same retrieval-token contract.

## Vision Backend Contract

Every CLIP-compatible vision backend should expose:

```python
encode_retrieval(image_tensor) -> Tensor  # shape (B, D)
```

For transformer-based lunar vision models, the backend should own:

```text
patchify image
infer target patch grid
resize absolute positional embeddings when needed
prepend retrieval token
run transformer
return token 0
```

This means the following details belong in `lunar_vision`, not `lunar_clip`:

```text
conv_proj
pos_embed
retrieval token
transformer module name
patch size
image size
variable-resolution support
```

GEO, WAC, and Fusion expose the same method:

```python
geo_model.encode_retrieval(geomap_tensor)
wac_model.encode_retrieval(wac_tensor)
fusion_model.encode_retrieval(wac_tensor)
```

All methods should return:

```text
(batch_size, output_dim)
```

The universal vision pooling policy should be retrieval-token pooling. Mean
pooling should not be the default GEO CLIP strategy.

## CLIP Adapter Contract

`LunarVisionEncoder` is now a thin adapter:

```text
load selected backend
validate modality
extract the correct tensor from the batch
apply minimal input preparation
call backend.encode_retrieval(tensor)
return RetrievalBatch
```

It should not directly resize positional embeddings or call backend internals.

Likewise, `LunarTextEncoder` should mostly:

```text
load selected text backend
tokenize or pass through token batch
call backend retrieval path
return RetrievalBatch
```

## Training And Evaluation

Training uses symmetric multi-positive contrastive loss over normalized CLIP
embeddings. Patch IDs identify positives, allowing multiple captions to match
one image; the model retains a one-to-one fallback for direct API calls without
IDs.

Metric names should distinguish:

```text
in_batch_text_to_image_top1
in_batch_image_to_text_top1
in_batch_retrieval_top1

full_text_to_image_recall@1/@5/@10
full_image_to_text_recall@1/@5/@10
full_text_to_image_median_rank
full_image_to_text_median_rank
full_text_to_image_mean_rank
full_image_to_text_mean_rank
full_mean_recall
```

In-batch metrics are useful during training. Full-dataset retrieval is a
separate evaluation path that compares queries against the complete validation
or test embedding index, including per-caption-version text-to-image metrics.
Final model comparison should use the `full_*` metrics, not the `in_batch_*`
diagnostics.

## Remaining Checkpoint Requirements

Every CLIP training run should eventually save enough configuration in or next
to the checkpoint to reconstruct the full model and adapter setup. Evaluation
JSON artifacts already retain model, dataset, and training metadata, but the
checkpoint itself currently stores only the training configuration.

The saved run config should include:

```text
text encoder name
tokenizer path
text checkpoint path
text freeze setting
vision encoder name
vision checkpoint path
vision freeze setting
modality
projection dimension
temperature
data paths
patch size and stride
training settings
```

The checkpoint or run directory should make it unambiguous which tokenizer,
encoder checkpoints, modality, and projection settings were used.

## Implemented Core

The implemented code shape is:

```python
text_vectors = text_backend.encode_retrieval(text_batch)
image_vectors = vision_backend.encode_retrieval(image_tensor)

text_embeds = normalize(text_projection(text_vectors))
image_embeds = normalize(image_projection(image_vectors))
```

That core LunarCLIP shape is implemented: domain backends return retrieval
vectors, while CLIP owns projection, normalization, similarity logits, and
contrastive alignment. Remaining work is primarily CLI coverage, fully
reconstructable checkpoints, populated non-BPE run configs, and removal of the
last modality/freeze-policy coupling from the CLIP adapter.
