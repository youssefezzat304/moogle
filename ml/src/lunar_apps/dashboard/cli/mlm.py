from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from lunar_apps.dashboard.config import CONFIG


MLM_TRAINING_DEFAULTS = (CONFIG.get("mlm") or {}).get("training") or {}
DEFAULT_MLM_SPLIT_DIR = "data/mlm/v1.0"


def add_mlm_subcommands(subparsers) -> None:
    parser = subparsers.add_parser(
        "mlm",
        help="Masked language model workflows.",
    )
    mlm_subparsers = parser.add_subparsers(
        dest="mlm_command",
        required=True,
    )

    split_parser = mlm_subparsers.add_parser(
        "split",
        help="Create train/eval/test parquet splits for MLM training.",
    )
    split_parser.add_argument(
        "--dataset-path",
        type=str,
        default="results/v4.0/combined_captions.parquet",
        help="Path to the combined long-format captions parquet.",
    )
    split_parser.add_argument(
        "--saved-path",
        type=str,
        default=DEFAULT_MLM_SPLIT_DIR,
        help="Folder where train/eval/test parquet files and config.json are saved.",
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of patch IDs assigned to the training split.",
    )
    split_parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of patch IDs assigned to the evaluation split.",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle patch IDs before splitting.",
    )
    split_parser.set_defaults(handler=run_split_command)

    inspect_parser = mlm_subparsers.add_parser(
        "inspect-batch",
        help="Inspect one tokenized and masked MLM batch.",
    )
    inspect_parser.add_argument(
        "--dataset-path",
        type=str,
        default=MLM_TRAINING_DEFAULTS.get(
            "train_path",
            f"{DEFAULT_MLM_SPLIT_DIR}/train.parquet",
        ),
        help="Path to a prepared MLM split parquet.",
    )
    inspect_parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=MLM_TRAINING_DEFAULTS.get(
            "tokenizer_path",
            "artifacts/tokenizers/bpe/v4.0/tokenizer.json",
        ),
        help="Path to the BPE tokenizer JSON.",
    )
    inspect_parser.add_argument(
        "--batch-size",
        type=int,
        default=MLM_TRAINING_DEFAULTS.get("batch_size", 4),
        help="Number of examples to collate for inspection.",
    )
    inspect_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MLM_TRAINING_DEFAULTS.get("max_seq_len", 128),
        help="Maximum token sequence length for the tokenizer wrapper.",
    )
    inspect_parser.add_argument(
        "--preview-tokens",
        type=int,
        default=32,
        help="Number of token IDs to print from the first example.",
    )
    inspect_parser.set_defaults(handler=run_inspect_batch_command)

    train_parser = mlm_subparsers.add_parser(
        "train",
        help="Train the BPE masked language model.",
    )
    train_parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to the prepared MLM training parquet split.",
    )
    train_parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Path to the prepared MLM evaluation parquet split.",
    )
    train_parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to the BPE tokenizer JSON.",
    )
    train_parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory where run folders are created.",
    )
    train_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Human-readable run name appended to the timestamped run folder.",
    )
    train_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum token sequence length for MLM training.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training and evaluation batch size.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="AdamW learning rate.",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay.",
    )
    train_parser.add_argument(
        "--disable-lr-scheduler",
        action="store_true",
        help="Disable the linear warmup plus cosine decay learning-rate scheduler.",
    )
    train_parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=None,
        help="Optimizer steps used for linear LR warmup. Overrides --lr-warmup-ratio when greater than 0.",
    )
    train_parser.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=None,
        help="Fraction of total optimizer steps used for linear LR warmup.",
    )
    train_parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=None,
        help="Final LR as a fraction of the base LR after cosine decay.",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum optimizer steps. Use 0 to train for all epochs.",
    )
    train_parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Run evaluation every N optimizer steps.",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Append training metrics every N optimizer steps.",
    )
    train_parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save a checkpoint every N optimizer steps.",
    )
    train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping max norm. Use 0 to disable clipping.",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on: auto, cpu, cuda, or another torch device string.",
    )
    train_parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="Maximum eval batches per eval pass. Use 0 for the full eval split.",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch random seed for reproducible smoke runs.",
    )
    train_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume MLM training from.",
    )
    train_parser.set_defaults(handler=run_train_command)

    tensorboard_parser = mlm_subparsers.add_parser(
        "export-tensorboard",
        help="Backfill TensorBoard event files from existing MLM metrics.csv runs.",
    )
    tensorboard_parser.add_argument(
        "--runs-root",
        type=str,
        default=MLM_TRAINING_DEFAULTS.get("output_root", "results/mlm/runs"),
        help="Root directory containing MLM run folders.",
    )
    tensorboard_parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite TensorBoard event files even when a run already has events.",
    )
    tensorboard_parser.set_defaults(handler=run_export_tensorboard_command)


def run_split_command(args, console: Console) -> None:
    from lunar_text.training.bpe.split import create_and_save_mlm_splits

    config = create_and_save_mlm_splits(
        dataset_path=args.dataset_path,
        saved_path=args.saved_path,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    summary = config["summary"]
    console.print(
        Panel.fit(
            "[bold blue]Created MLM Splits[/bold blue]\n"
            f"Input  : {config['input_path']}\n"
            f"Output : {config['output_dir']}\n"
            f"Seed   : {config['seed']}\n\n"
            f"Train  : {summary['train']['rows']} rows, "
            f"{summary['train']['patch_ids']} patches\n"
            f"Eval   : {summary['eval']['rows']} rows, "
            f"{summary['eval']['patch_ids']} patches\n"
            f"Test   : {summary['test']['rows']} rows, "
            f"{summary['test']['patch_ids']} patches",
            title="LunarGeo MLM",
        )
    )


def run_inspect_batch_command(args, console: Console) -> None:
    from torch.utils.data import DataLoader

    from lunar_text.model.bpe.config import ModelConfig
    from lunar_text.training.bpe.dataset import MLMDataset, build_dataset
    from lunar_text.utils.tokenizers import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer_path)
    config = ModelConfig(
        max_seq_len=args.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("[PAD]"),
    )
    dataset = build_dataset(
        dataset_path=args.dataset_path,
        config=config,
        tokenizer_path=args.tokenizer_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MLMDataset.collate,
    )
    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    first_input = input_ids[0]
    first_labels = labels[0]
    active_positions = first_labels != -100
    active_label_ids = first_labels[active_positions].tolist()
    active_positions_list = active_positions.nonzero(as_tuple=False).flatten().tolist()
    preview = min(args.preview_tokens, first_input.numel())

    decoded = dataset.wrapper.decode(first_input, skip_special_tokens=False)
    console.print(
        Panel.fit(
            "[bold blue]MLM Batch Inspection[/bold blue]\n"
            f"Dataset       : {args.dataset_path}\n"
            f"Tokenizer     : {args.tokenizer_path}\n"
            f"Dataset rows  : {len(dataset)}\n"
            f"Batch size    : {input_ids.shape[0]}\n"
            f"Sequence len  : {input_ids.shape[1]}\n"
            f"Masked tokens : {int((labels != -100).sum().item())}\n"
            f"Real tokens   : {int(attention_mask.sum().item())}\n\n"
            f"Input IDs     : {first_input[:preview].tolist()}\n"
            f"Label positions: {active_positions_list[:preview]}\n"
            f"Label IDs     : {active_label_ids[:preview]}\n"
            f"Decoded first : {decoded}",
            title="LunarGeo MLM",
        )
    )


def run_train_command(args, console: Console) -> None:
    from lunar_text.training.bpe.trainer import train_mlm

    config = _build_training_config_from_args(args)

    console.print(
        Panel.fit(
            "[bold blue]Starting MLM Training[/bold blue]\n"
            f"Train     : {config.train_path}\n"
            f"Eval      : {config.eval_path}\n"
            f"Tokenizer : {config.tokenizer_path}\n"
            f"Run name  : {config.run_name}\n"
            f"Max steps : {config.max_steps}\n"
            f"Resume    : {config.resume_checkpoint or '-'}",
            title="LunarGeo MLM",
        )
    )

    result = train_mlm(config, console=console)

    console.print(
        Panel.fit(
            "[bold green]MLM Training Complete[/bold green]\n"
            f"Run dir     : {result.run_dir}\n"
            f"Metrics     : {result.metrics_path}\n"
            f"Checkpoint  : {result.final_checkpoint_path}\n"
            f"Global step : {result.global_step}\n"
            f"Latest      : {result.latest_metrics}",
            title="LunarGeo MLM",
        )
    )


def run_export_tensorboard_command(args, console: Console) -> None:
    from lunar_text.training.bpe.tensorboard_export import export_metrics_csv_to_tensorboard

    exported = export_metrics_csv_to_tensorboard(
        runs_root=args.runs_root,
        force=args.force,
    )

    if not exported:
        console.print(
            Panel.fit(
                "[yellow]No MLM runs needed TensorBoard export.[/yellow]\n"
                f"Runs root: {args.runs_root}",
                title="LunarGeo MLM",
            )
        )
        return

    console.print(
        Panel.fit(
            "[bold green]TensorBoard export complete.[/bold green]\n"
            f"Runs root : {args.runs_root}\n"
            f"Exported  : {len(exported)} run(s)\n\n"
            "Start TensorBoard with:\n"
            f"uv run tensorboard --logdir {args.runs_root}",
            title="LunarGeo MLM",
        )
    )


def _build_training_config_from_args(args) -> MLMTrainingConfig:
    from lunar_text.training.bpe.trainer import MLMTrainingConfig

    defaults = MLMTrainingConfig()
    return MLMTrainingConfig(
        train_path=_arg_or_config(args.train_path, "train_path", defaults.train_path),
        eval_path=_arg_or_config(args.eval_path, "eval_path", defaults.eval_path),
        tokenizer_path=_arg_or_config(
            args.tokenizer_path,
            "tokenizer_path",
            defaults.tokenizer_path,
        ),
        output_root=_arg_or_config(
            args.output_root, "output_root", defaults.output_root
        ),
        run_name=_arg_or_config(args.run_name, "run_name", defaults.run_name),
        max_seq_len=_arg_or_config(
            args.max_seq_len, "max_seq_len", defaults.max_seq_len
        ),
        batch_size=_arg_or_config(args.batch_size, "batch_size", defaults.batch_size),
        epochs=_arg_or_config(args.epochs, "epochs", defaults.epochs),
        learning_rate=_arg_or_config(
            args.learning_rate,
            "learning_rate",
            defaults.learning_rate,
        ),
        weight_decay=_arg_or_config(
            args.weight_decay,
            "weight_decay",
            defaults.weight_decay,
        ),
        use_lr_scheduler=(
            False
            if args.disable_lr_scheduler
            else _arg_or_config(
                None,
                "use_lr_scheduler",
                defaults.use_lr_scheduler,
            )
        ),
        lr_warmup_steps=_arg_or_config(
            args.lr_warmup_steps,
            "lr_warmup_steps",
            defaults.lr_warmup_steps,
        ),
        lr_warmup_ratio=_arg_or_config(
            args.lr_warmup_ratio,
            "lr_warmup_ratio",
            defaults.lr_warmup_ratio,
        ),
        lr_min_ratio=_arg_or_config(
            args.lr_min_ratio,
            "lr_min_ratio",
            defaults.lr_min_ratio,
        ),
        max_steps=_arg_or_config(args.max_steps, "max_steps", defaults.max_steps),
        eval_every=_arg_or_config(args.eval_every, "eval_every", defaults.eval_every),
        log_every=_arg_or_config(args.log_every, "log_every", defaults.log_every),
        save_every=_arg_or_config(args.save_every, "save_every", defaults.save_every),
        grad_clip=_arg_or_config(args.grad_clip, "grad_clip", defaults.grad_clip),
        device=_arg_or_config(args.device, "device", defaults.device),
        eval_max_batches=_arg_or_config(
            args.eval_max_batches,
            "eval_max_batches",
            defaults.eval_max_batches,
        ),
        num_workers=_arg_or_config(
            args.num_workers, "num_workers", defaults.num_workers
        ),
        seed=_arg_or_config(args.seed, "seed", defaults.seed),
        resume_checkpoint=args.resume_checkpoint,
    )


def _arg_or_config(value, config_key: str, fallback):
    if value is not None:
        return value
    return MLM_TRAINING_DEFAULTS.get(config_key, fallback)
