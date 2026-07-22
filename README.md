# Python

> [!IMPORTANT]
> **Git LFS Required** — This repository uses [Git Large File
> Storage](https://git-lfs.com/) for large binary files. You **must** set it up
> before working with this repo, otherwise you will only get pointer files.
>
> ```sh
> # Install the Git LFS hooks (required once per machine)
> git lfs install
> ```
>
> After cloning, pull all LFS-tracked files:
>
> ```sh
> git lfs pull
> ```
>
> The included `.gitattributes` file tracks common large file types (model
> checkpoints, pickled data, images, etc.). If you need to track additional
> file types, add them with:
>
> ```sh
> git lfs track "*.<extension>"
> ```
>
> Replace `<extension>` with the desired file type to be tracked, like `mat` or `png`.

## Project Structure

This template uses [uv](https://docs.astral.sh/uv/) with
[`uv_build`](https://docs.astral.sh/uv/guides/package/) as the default build
backend (configured in `pyproject.toml`).

- **`src/`** — Reusable library code: functions, classes, and modules that your
  scripts will import from. This is what `uv_build` packages for distribution.
  **Do not put scripts here** — use the `scripts/` folder instead.
- **`scripts/`** — Entry points, experiments, one-off utilities, and any `.py`
  files that _use_ your library code from `src/`. These are not packaged.
- **`test/`** — Test files for your package (pytest discovers `test/test_*.py`).

### Example Layout

Imagine a project that trains models and visualizes results:

```
my_project/
├── pyproject.toml
├── src/
│   ├── data/            # Data loading & preprocessing
│   │   └── __init__.py
│   ├── models/          # Model definitions
│   │   └── __init__.py
│   └── viz/             # Visualization utilities
│       └── __init__.py
├── scripts/
│   ├── train.py         # imports from src.data and src.models
│   └── visualize.py     # imports from src.viz
└── test/
    └── test_models.py
```

The `scripts/` files import from `src/` like a normal library:

```python
# scripts/train.py
from data import load_dataset
from models import MyModel
...
```

```python
# scripts/visualize.py
from viz import plot_results
...
```

### Multiple Modules

By default, `uv_build` expects a single module matching the project name
(`src/<package_name>/`). If your project has multiple independent packages under
`src/`, list them in `pyproject.toml`:

```toml
[tool.uv.build-backend]
module-name = ["data", "models", "viz"]
```

For more complex layouts (namespace packages, etc.), see the
[uv build backend docs](https://docs.astral.sh/uv/concepts/build-backend/#modules).

## Renaming the Package

Before you start, rename the package from `example_package` to your own project
name:

1. Rename the directory `src/example_package/` to `src/<your_package>/`
2. Update the `name` field in `pyproject.toml` to match

## UV

If you are using Python for your project, please use [uv](https://docs.astral.sh/uv/) to manage your packages and (virtual-)environments. This will make it easier to create an environment with all the needed packages in there. While other options do exist (like poetry), they do not follow the Python PEP standard and uv is the only tool to do so.

- Initialize a project: `uv init`
- Add packages to `pyproject.toml` and to the environment: `uv add <package-name>`,
  for example `uv add numpy`
- Remove package: `uv remove <package-name>`
- If pulling changes from git and you want your environment to be synchronized (install/remove new/old packages): `uv sync`
  Running files: `uv run <file>.py`, for example `uv run scripts/main.py`
- You can also "activate" the environment like it is usually done with `source .venv/bin/activate` or any other activation script found in `.venv/bin`

## Testing

Run tests with [pytest](https://docs.pytest.org/):

```sh
uv run pytest
```

For coverage reports (pytest-cov is included in the test dependency group):

```sh
uv run pytest --cov=src/example_package
```

## Optional Pre-commit Hooks

This template includes an optional
[pre-commit](https://pre-commit.com/) configuration that runs basic file checks
and Ruff before each commit. It is opt-in: the hooks only run after you install
and enable pre-commit locally.

[Ruff](https://docs.astral.sh/ruff/) is a fast Python code formatter and linter.
This template uses two Ruff commands:

- `ruff format` formats Python files automatically so code style stays
  consistent.
- `ruff check` finds common mistakes, unused imports, style issues, and other
  lint problems. In the pre-commit hook, it runs with `--fix` so Ruff can repair
  safe issues automatically.

Install pre-commit as a uv tool:

```sh
uv tool install pre-commit
```

Enable the Git hook for this repository:

```sh
pre-commit install
```

You can also run all hooks manually:

```sh
pre-commit run --all-files
```

## PyTorch with GPU Acceleration

The template includes commented-out PyTorch GPU configurations in
`pyproject.toml` with pre-configured indexes for three backends:

- **NVIDIA CUDA** (`cu124`) — For NVIDIA GPUs. Adjust the CUDA version (e.g.
  `cu124`, `cu126`) to match your target machine.
- **Intel Arc / XPU** (`xpu`) — For Intel Arc GPUs found in many modern laptops.
  The `xpu` index provides PyTorch builds with Intel GPU support.
- **AMD ROCm** (`rocm7.2`) — For AMD GPUs. Adjust the ROCm version (e.g.
  `rocm6.2`, `rocm7.2`) to match your system.

Browse available versions at
[download.pytorch.org/whl](https://download.pytorch.org/whl/).

### Setup

1. Uncomment the `[[tool.uv.index]]` block for your GPU in `pyproject.toml`
2. Uncomment `[tool.uv.sources]` and point `torch`, `torchvision`, `torchaudio`
   to your chosen index
3. Install the packages using `uv add`:

```sh
uv add torch torchvision torchaudio
```

### PyTorch Geometric (PyG)

If you need PyTorch Geometric and its sparse extensions, uncomment:

- `[tool.uv]` → `find-links = ["https://data.pyg.org/whl/"]`
- `[tool.uv.extra-build-dependencies]` for `torch-scatter`, `torch-sparse`,
  `torch-cluster`

Then install:

```sh
uv add torch-geometric torch-scatter torch-sparse torch-cluster
```

## Commits

Please follow the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for your messages.
