import argparse
import sys
from pathlib import Path

from rich.console import Console

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lunar_apps.dashboard.cli.catalog import add_catalog_subcommands
from lunar_apps.dashboard.cli.clip import add_clip_subcommands
from lunar_apps.dashboard.cli.generation import add_generation_subcommand
from lunar_apps.dashboard.cli.index import add_index_subcommands
from lunar_apps.dashboard.cli.mlm import add_mlm_subcommands
from lunar_apps.dashboard.cli.tokenizer import add_tokenizer_subcommands

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LunarGeo data, tokenizer, MLM, and CLIP workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_generation_subcommand(subparsers)
    add_tokenizer_subcommands(subparsers)
    add_mlm_subcommands(subparsers)
    add_clip_subcommands(subparsers)
    add_catalog_subcommands(subparsers)
    add_index_subcommands(subparsers)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args, console)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
