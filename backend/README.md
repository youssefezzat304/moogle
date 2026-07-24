# Moogle Backend

The backend owns Moogle's HTTP contracts, request validation, API routes, and
orchestration. Model loading and similarity search remain owned by
`../inference`.

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

Start the API after the production catalog and index have been built:

```sh
uv run --project backend python backend/main.py
```

The MVP exposes:

```text
POST /api/retrieval
GET  /api/patches/{patch_id}/wac
```
