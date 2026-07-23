# Moogle Backend

The backend owns Moogle's HTTP contracts, request validation, API routes, and
orchestration. LunarCLIP model loading, indexing, and similarity search remain
owned by `../ml`.

The authoritative retrieval wire contract is
[`../docs/api/retrieval-v1.yaml`](../docs/api/retrieval-v1.yaml). Backend and
frontend contract tests consume the shared fixture under `../tests/fixtures/`.

## Development

From the repository root:

```sh
uv sync --project backend --dev
uv run --project backend pytest
uv run --project backend ruff check .
uv run --project backend ruff format --check .
```
