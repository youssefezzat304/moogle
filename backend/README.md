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
uv run --project backend pytest backend/tests
uv run --project backend ruff check backend
uv run --project backend ruff format --check backend
```

Start the API after the production catalog and index have been built:

```sh
cp backend/.env.example backend/.env
uv run --env-file backend/.env --project backend python backend/main.py
```

Artifact paths in the environment file may be absolute or relative to the
repository root. The default runtime uses CPU-only PyTorch.

The MVP exposes:

```text
GET  /api/health
POST /api/retrieval
GET  /api/patches/{patch_id}/wac
```
