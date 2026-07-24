from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from lunar_data.catalog import build_catalog_from_recipe, validate_catalog_artifact


DATA_ROOT = Path("/home/pg2026/data")
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def add_catalog_subcommands(subparsers) -> None:
    parser = subparsers.add_parser(
        "catalog",
        help="Build canonical lunar catalog artifacts.",
    )
    catalog_subparsers = parser.add_subparsers(
        dest="catalog_command",
        required=True,
    )

    build_parser = catalog_subparsers.add_parser(
        "build",
        help="Build and validate a catalog from a YAML recipe.",
    )
    build_parser.add_argument(
        "--config",
        required=True,
        help="Path to the catalog YAML recipe.",
    )
    build_parser.set_defaults(handler=run_catalog_build_command)


def run_catalog_build_command(args, console: Console) -> None:
    config_path = Path(args.config)
    console.print(f"[cyan]Catalog config:[/cyan] {config_path}")
    console.print(f"[cyan]Lunar data root:[/cyan] {DATA_ROOT}")
    console.print("[cyan]Preflighting inputs and building catalog...[/cyan]")

    output = build_catalog_from_recipe(
        config_path,
        repository_root=REPOSITORY_ROOT,
        data_root=DATA_ROOT,
    )
    artifact = validate_catalog_artifact(output)
    console.print(
        Panel.fit(
            "[bold green]Catalog build complete[/bold green]\n"
            f"Catalog : {artifact.manifest.catalog_id}\n"
            f"Rows    : {artifact.metadata.height}\n"
            f"Output  : {output}",
            title="Lunar Catalog",
        )
    )
