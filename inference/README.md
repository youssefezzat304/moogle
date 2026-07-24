# Moogle Inference

The inference package loads the promoted LunarCLIP model, canonical catalog,
and model-specific image index into one in-memory retrieval engine.

```python
from moogle_inference import load_retrieval_engine

engine = load_retrieval_engine(
    catalog_path="storage/catalogs/lunar-v1",
    index_path="storage/indexes/bpe_geo/v1",
    model_manifest_path="storage/models/bpe_geo/manifest.yaml",
    device="cuda",
)

results = engine.search("young crater with bright ejecta", top_k=5)
```

Each result contains its rank, raw cosine similarity, canonical patch ID,
description, caption provenance, coordinates, and local WAC image path. HTTP
routing and conversion of the image path to an API URL belong to the backend.
