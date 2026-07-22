# Moogle Frontend

The Moogle frontend is a React application for exploring mock lunar retrieval
results in an interactive 3D interface. It is built with TypeScript, Vite,
Tailwind CSS, React Three Fiber, and Framer Motion.

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
npm run format        # Format supported files with Prettier
npm run format:check  # Check formatting without modifying files
```

## Architecture

The application uses a feature-oriented structure under `src/`:

- `app/` contains the React entry point and top-level application composition.
- `features/chat/` implements the query and retrieval conversation interface.
- `features/moon/` contains the Three.js lunar visualization and canvas setup.
- `features/retrieval/` defines the mock retrieval data and matching behavior.
- `shared/` contains reusable layout components and hooks.
- `styles/` contains global styles and application-specific CSS.
- `public/` contains lunar imagery and other static assets served by Vite.

The `@/` alias resolves to `src/`. Tailwind is loaded through `src/styles/index.css`
and integrated into Vite by `@tailwindcss/vite`.

## Environment variables

No environment variables are currently required; retrieval data is local mock
data. For future configuration, create an untracked `.env.local` file in this
directory. Vite only exposes variables prefixed with `VITE_` to browser code:

```dotenv
# Example for a future API integration; this is not currently consumed.
VITE_API_BASE_URL=http://localhost:8000
```

Read a client variable with `import.meta.env.VITE_API_BASE_URL` and restart the
development server after changing an environment file. Never put secrets in a
`VITE_` variable because its value is included in the browser bundle.
