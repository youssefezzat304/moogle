from rich.console import Console
from rich.panel import Panel


def add_tokenizer_subcommands(subparsers) -> None:
    parser = subparsers.add_parser(
        "tokenizer",
        help="Tokenizer training workflows.",
    )
    tokenizer_subparsers = parser.add_subparsers(
        dest="tokenizer_command",
        required=True,
    )

    bpe_parser = tokenizer_subparsers.add_parser(
        "train-bpe",
        help="Train a BPE tokenizer.",
    )
    bpe_parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/v1.0/train.parquet",
        help="Path to the parquet or text file used for tokenizer training.",
    )
    bpe_parser.add_argument(
        "--output",
        "--t-output",
        dest="output",
        type=str,
        default="artifacts/tokenizers/bpe/v4.0/tokenizer.json",
        help="Output path for the trained tokenizer.",
    )
    bpe_parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column to use when training from parquet.",
    )
    bpe_parser.add_argument(
        "--vocab-size",
        type=int,
        default=100000,
        help="Tokenizer vocab size.",
    )
    bpe_parser.set_defaults(handler=run_train_bpe_command)


def run_train_bpe_command(args, console: Console) -> None:
    from lunar_text.tokenizer.bpe.train import train_bpe_tokenizer

    console.print(
        Panel.fit(
            "[bold blue]Training BPE Tokenizer[/bold blue]\n"
            f"Dataset : {args.dataset_path}\n"
            f"Output  : {args.output}\n"
            f"Column  : {args.text_column}\n"
            f"Vocab   : {args.vocab_size} tokens",
            title="LunarGeo Tokenizer",
        )
    )
    train_bpe_tokenizer(
        dataset_path=args.dataset_path,
        output_path=args.output,
        vocab_size=args.vocab_size,
        text_column=args.text_column,
    )
    console.print("[bold green]Tokenizer training complete.[/bold green]")
