from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from lunar_apps.dashboard.config import CONFIG
from lunar_apps.dashboard.prompts import PROMPT_VERSION


def add_generation_subcommand(subparsers) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Generate lunar patch descriptions.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with safe test defaults.",
    )
    parser.add_argument(
        "--dir",
        "--output-dir",
        dest="dir",
        type=str,
        default=f"results/{PROMPT_VERSION}",
        help="Output directory for the generation pipeline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of random patches to process. Omit for full production run.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start padding index.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=0,
        help="End padding index.",
    )
    parser.set_defaults(handler=run_generation_command)


def run_generation_command(args, console: Console) -> None:
    from lunar_apps.dashboard.descriptions_generator import run_generation_pipeline

    if args.dev:
        console.print(
            Panel.fit(
                "[bold yellow]Running in DEVELOPMENT mode[/bold yellow]\n"
                "Using safe test defaults.",
                title="LunarGeo Pipeline",
            )
        )
        limit = 20
        start_padding = 50
        end_padding = 50
    else:
        console.print(
            Panel.fit(
                "[bold green]Running in PRODUCTION mode[/bold green]\n"
                "Full pipeline execution enabled.",
                title="LunarGeo Pipeline",
            )
        )
        limit = args.limit
        start_padding = args.start
        end_padding = args.end

    console.print(f"[cyan]Output directory:[/cyan] {args.dir}/")
    console.print(f"[cyan]Patch size:[/cyan] {CONFIG['hyperparameters']['patch_size']}px")
    console.print(f"[cyan]Stride:[/cyan] {CONFIG['hyperparameters']['stride']}px")

    run_generation_pipeline(
        limit=limit,
        start_padding=start_padding,
        end_padding=end_padding,
        output_dir=Path(args.dir),
    )
