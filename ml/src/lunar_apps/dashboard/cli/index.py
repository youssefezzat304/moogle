from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from lunar_clip.retrieval.indexing import (
    build_index_from_recipe,
    load_index_artifact,
)


DATA_ROOT = Path("/home/pg2026/data")
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def add_index_subcommands(subparsers) -> None:
    parser = subparsers.add_parser(
        "index",
        help="Build model-specific lunar embedding indexes.",
    )
    index_subparsers = parser.add_subparsers(
        dest="index_command",
        required=True,
    )

    build_parser = index_subparsers.add_parser(
        "build",
        help="Build and validate an embedding index from a YAML recipe.",
    )
    build_parser.add_argument(
        "--config",
        required=True,
        help="Path to the embedding-index YAML recipe.",
    )
    build_parser.set_defaults(handler=run_index_build_command)


def run_index_build_command(args, console: Console) -> None:
    config_path = Path(args.config)
    console.print(f"[cyan]Index config:[/cyan] {config_path}")
    console.print(f"[cyan]Lunar data root:[/cyan] {DATA_ROOT}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Encoding geomap patches", total=None)

        def report(completed: int, total: int) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=total or None,
            )

        output = build_index_from_recipe(
            config_path,
            repository_root=REPOSITORY_ROOT,
            data_root=DATA_ROOT,
            progress=report,
        )

    artifact = load_index_artifact(output)
    console.print(
        Panel.fit(
            "[bold green]Embedding index build complete[/bold green]\n"
            f"Index   : {artifact.manifest.index_id}\n"
            f"Catalog : {artifact.manifest.catalog_id}\n"
            f"Model   : {artifact.manifest.descriptor.model_id}\n"
            f"Shape   : {tuple(artifact.embeddings.shape)}\n"
            f"Output  : {output}",
            title="Lunar Index",
        )
    )
