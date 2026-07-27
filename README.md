# Moogle

Moogle is a semantic search engine for lunar terrain. It encodes a text query
with LunarCLIP, searches a prebuilt geomap embedding index, and displays the
best-matching lunar patches using corresponding WAC imagery.

## Build retrieval artifacts

With the lunar source data under `/home/pg2026/data`, build the catalog once:

```sh
cd ml
uv run python main.py catalog build --config configs/catalog/lunar-v1.yaml
```

Then build the `bpe_geo` embedding index on a CUDA-capable machine:

```sh
uv run python main.py index build --config configs/index/bpe_geo-v1.yaml
cd ..
```

These commands create `storage/catalogs/lunar-v1/` and
`storage/indexes/bpe_geo/v1/`.

## Run locally

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22, and the
model, catalog, and index artifacts under `storage/`.

```sh
uv sync
npm --prefix frontend install
cp backend/.env.example backend/.env
./scripts/run-dev
```

Open `http://localhost:5173`. The backend API runs at
`http://127.0.0.1:8000`. Press `Ctrl+C` to stop both services.
