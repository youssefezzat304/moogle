# Moogle Frontend

The Moogle frontend is a React application for exploring lunar retrieval
results in an interactive 3D interface. It is built with TypeScript, Vite,
Tailwind CSS, React Three Fiber, and Framer Motion. Retrieval results come only
from the configured API; the production interface has no landmark fallback.

## Requirements

- Node.js 22.12 or newer
- npm 10.5.1 or newer

These versions satisfy the runtime requirements of Vite and the frontend's
current dependencies. A Node version manager such as `nvm`, `fnm`, or `mise` is
recommended.

## Setup

From the repository root:

```sh
cd frontend
npm install
npm run dev
```

Vite prints the local development URL when the server starts. Other useful
commands are:

```sh
npm run build         # Type-check and create a production build
npm run preview       # Serve the production build locally
npm run lint          # Run ESLint
npm run test          # Run unit tests
npm run format        # Format supported files with Prettier
npm run format:check  # Check formatting without modifying files
```

## Architecture

The application uses a feature-oriented structure under `src/`:

- `app/` contains the React entry point and top-level application composition.
- `features/search/` owns search state and the ranked retrieval interface.
- `features/moon/` contains the Three.js lunar visualization and canvas setup.
- `features/retrieval/` defines the API contract, response validation, and
  retrieval client.
- `shared/` contains reusable layout components and hooks.
- `styles/` contains global styles and application-specific CSS.
- `public/` contains lunar imagery and other static assets served by Vite.

The `@/` alias resolves to `src/`. Tailwind is loaded through `src/styles/index.css`
and integrated into Vite by `@tailwindcss/vite`.

## Retrieval API

The frontend sends `POST /api/retrieval` by default. During local development,
Vite proxies `/api` to `http://localhost:8000`. The authoritative wire contract
is [`docs/api/retrieval-v1.yaml`](../docs/api/retrieval-v1.yaml). This repository
does not yet provide the endpoint, so queries fail visibly until a compatible
service is running.

The request body is:

```json
{
  "query": "bright ejecta around a fresh crater",
  "top_k": 5
}
```

The response contract is:

```json
{
  "schema_version": 1,
  "query": "bright ejecta around a fresh crater",
  "model_id": "bpe_geo",
  "index_size": 22578,
  "elapsed_ms": 84,
  "results": [
    {
      "rank": 1,
      "patch_id": 1234,
      "image_url": "/api/patches/1234/image",
      "latitude": -12.34,
      "longitude": 45.67,
      "similarity": 0.312,
      "description": "A fresh crater surrounded by bright ejecta material.",
      "source_version": "v2.0",
      "prompt_style": "llm_description"
    }
  ]
}
```

Similarity is displayed as the raw model value. The frontend does not convert
it into a confidence percentage.

## Environment variables

Create an untracked `.env.local` file to override either API location:

```dotenv
# Browser-facing base URL. Defaults to /api.
VITE_API_BASE_URL=/api

# Vite development proxy target. Defaults to http://localhost:8000.
VITE_API_PROXY_TARGET=http://localhost:8000
```

Restart the development server after changing an environment file. Never put
secrets in a `VITE_` variable because its value is included in the browser
bundle.
