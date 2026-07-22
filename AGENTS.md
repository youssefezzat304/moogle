# AGENTS.md

Instructions for AI coding assistants working on this project.

## Project Overview

This is a Python project template using [uv](https://docs.astral.sh/uv/) for package
management and [uv_build](https://docs.astral.sh/uv/guides/package/) as the build
backend. The project follows a `src`-based layout with separate directories for
library code, scripts, and tests.

## Environment & Devenv

**If `devenv` is available on the system**, all commands must be prefixed with
`devenv shell --`. For example:

```sh
devenv shell -- uv run pytest
devenv shell -- uv add numpy
devenv shell -- uv run scripts/main.py
```

If `devenv` is not available, run commands directly (e.g. `uv run pytest`).

## Package Management — CRITICAL

- **Never** edit `dependencies` or `dependency-groups` fields in `pyproject.toml`
  directly. Doing so bypasses `uv`'s lockfile and version resolution, which can
  silently cause incorrect or incompatible versions.
- **Always** use `uv add <package>` to add dependencies and `uv add --dev <package>`
  for development dependencies.
- **Always** use `uv remove <package>` to remove dependencies.
- After pulling changes from git, run `uv sync`.
- **Everything else** in `pyproject.toml` (tool configs, project metadata, build
  settings) can be edited directly.

## Project Structure

```
src/
  example_package/   # Library/package code (rename to your project name)
scripts/             # Entry points and utility scripts that use the library
test/                # Tests (pytest discovers files matching test/test_*.py)
```

- Place reusable library/package code under `src/`.
- Place custom scripts that import from the library under `scripts/`.
- Tests go under `test/` and should be named `test_*.py`.

## Running & Testing

```sh
# Run a script
uv run scripts/main.py

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/example_package
```

## Type Checking

This project uses [pyright](https://github.com/microsoft/pyright):

```sh
uv run pyright
```

## Code Quality

This project uses [ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```sh
uv run ruff format
uv run ruff check
```

## Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## Renaming the Package

When starting a new project, rename the package from `example_package`:

1. Rename `src/example_package/` to `src/<your_package>/`
2. Update the `name` field in `pyproject.toml`
3. Update any imports that reference `example_package`
