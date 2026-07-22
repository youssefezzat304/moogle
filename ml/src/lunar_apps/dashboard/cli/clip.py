from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from lunar_vision.model.wac.config import WAC_NORMALIZE_MEAN, WAC_NORMALIZE_STD


def add_clip_subcommands(subparsers) -> None:
    parser = subparsers.add_parser(
        "clip",
        help="LunarCLIP training and encoding workflows.",
    )
    clip_subparsers = parser.add_subparsers(
        dest="clip_command",
        required=True,
    )

    train_parser = clip_subparsers.add_parser(
        "train",
        help="Train LunarCLIP from a YAML config.",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to the LunarCLIP YAML config.",
    )
    train_parser.set_defaults(handler=run_clip_train_command)

    text_parser = clip_subparsers.add_parser(
        "encode-text",
        help="Encode text with a LunarCLIP text encoder.",
    )
    text_parser.add_argument(
        "--encoder",
        default="bpe",
        choices=["bpe"],
        help="Text encoder backend.",
    )
    text_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the text checkpoint.",
    )
    text_parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the tokenizer JSON.",
    )
    text_parser.add_argument(
        "--text",
        required=True,
        help="Text to encode.",
    )
    text_parser.set_defaults(handler=run_clip_encode_text_command)

    image_parser = clip_subparsers.add_parser(
        "encode-image",
        help="Encode image patches with a CLIP vision adapter.",
    )
    image_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the vision checkpoint.",
    )
    image_parser.add_argument(
        "--modality",
        required=True,
        choices=["geomap", "wac"],
        help="Vision modality to encode.",
    )
    image_parser.add_argument(
        "--patch-id",
        required=True,
        type=int,
        help="Patch ID to encode.",
    )
    image_parser.set_defaults(handler=run_clip_encode_image_command)


def run_clip_train_command(args, console: Console) -> None:
    import yaml

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _print_clip_training_summary(
        console=console,
        config_path=args.config,
        config=config,
    )
    console.print("[cyan]Loading dataset, text encoder, and vision encoder...[/cyan]")
    (
        model,
        train_dataset,
        eval_dataset,
        training_config,
    ) = _build_clip_training_objects(config)

    from lunar_clip.training import train_clip

    result = train_clip(
        model=model,
        train_dataset=train_dataset,
        val_dataset=eval_dataset,
        config=training_config,
        evaluation_metadata=_build_retrieval_evaluation_metadata(config),
    )
    evaluation_artifacts = "\n".join(
        f"Evaluation artifact : {path}"
        for path in result.evaluation_artifact_paths
    ) or "Evaluation artifacts: none (no eval/test split available)"
    console.print(
        Panel.fit(
            "[bold green]LunarCLIP training complete[/bold green]\n"
            f"Best checkpoint : {result.best_model_path}\n"
            f"{evaluation_artifacts}",
            title="LunarCLIP",
        )
    )


def _print_clip_training_summary(
    console: Console,
    config_path: str,
    config: dict,
) -> None:
    experiment_config = config["experiment"]
    text_config = config["text"]
    vision_config = config["vision"]
    data_config = config["data"]
    training_config = config["training"]

    output_dir = Path(training_config.get("output_dir", "results/clip"))
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / training_config.get(
        "run_name",
        experiment_config["name"],
    )
    eval_fraction = float(data_config.get("eval_fraction", 0.0))

    console.print(
        Panel.fit(
            "[bold blue]LunarCLIP training run[/bold blue]\n"
            f"Config           : {config_path}\n"
            f"Experiment       : {experiment_config['name']}\n"
            f"Modality         : {experiment_config['modality']}\n"
            f"Text encoder     : {text_config['encoder']} "
            f"(freeze={text_config.get('freeze_encoder', False)})\n"
            f"Vision encoder   : {vision_config['encoder']} "
            f"(freeze={vision_config.get('freeze_encoder', False)})\n"
            f"Patch / stride   : {data_config['patch_size']} / {data_config['stride']}\n"
            f"Eval split       : {eval_fraction:.0%}\n"
            f"Epochs           : {training_config.get('epochs', 1)}\n"
            f"Batch size       : {training_config.get('batch_size', 32)}\n"
            f"Text encoder LR  : {training_config.get('text_encoder_learning_rate', training_config.get('learning_rate', 1e-4))}\n"
            f"Workers          : {training_config.get('num_workers', 0)}\n"
            f"Device config    : {training_config.get('accelerator', 'auto')}, "
            f"devices={training_config.get('devices', 'auto')}\n"
            f"Output dir       : {output_dir}\n"
            f"Checkpoints      : {checkpoint_dir}\n"
            f"TensorBoard logs : {tensorboard_dir}",
            title="LunarCLIP",
        )
    )


def run_clip_encode_text_command(args, console: Console) -> None:
    import torch

    from lunar_clip.encoders.text import LunarTextEncoder

    console.print(
        Panel.fit(
            "[bold blue]Encoding LunarCLIP text[/bold blue]\n"
            f"Checkpoint : {args.checkpoint}\n"
            f"Tokenizer  : {args.tokenizer}",
            title="LunarCLIP",
        )
    )
    adapter = LunarTextEncoder(
        encoder=args.encoder,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )
    adapter.eval()
    with torch.no_grad():
        encoded = adapter.encode_text(args.text)
    console.print(encoded.vectors.squeeze(0).detach().cpu().tolist())


def run_clip_encode_image_command(args, console: Console) -> None:
    del args
    raise RuntimeError(
        "Image encoding is exposed through LunarVisionEncoder. The CLI image "
        "encoding command still needs wiring for loading a patch by ID."
    )


def _build_clip_training_objects(config):
    from lunar_clip.data.clip_dataset import LunarCLIPDataset
    from lunar_clip.encoders.text import LunarTextEncoder
    from lunar_clip.model.lunar_clip_model import LunarCLIPModel
    from lunar_clip.training import LunarCLIPTrainingConfig
    from lunar_clip.utils import build_vision_adapter
    from lunar_data.lunar_geo_data import LunarGeoData

    experiment_config = config["experiment"]
    text_config = config["text"]
    vision_config = config["vision"]
    data_config = config["data"]
    model_config = config.get("model", {})
    training_config = config["training"]

    text_adapter = LunarTextEncoder(
        encoder=text_config["encoder"],
        tokenizer_path=text_config["tokenizer_path"],
        checkpoint_path=text_config["checkpoint_path"],
        freeze_encoder=text_config.get("freeze_encoder", False),
    )
    vision_adapter = build_vision_adapter(vision_config)

    modality = experiment_config["modality"]
    vision_dataset = LunarGeoData(
        root=data_config["vision_root"],
        patch_size=int(data_config["patch_size"]),
        stride=int(data_config["stride"]),
        transform=_vision_transforms(modality),
    )
    train_dataset = LunarCLIPDataset(
        captions_path=data_config["captions_path"],
        vision_dataset=vision_dataset,
        modality=modality,
        caption_policy=data_config.get("caption_policy", "sample_one"),
        split="train",
        seed=int(training_config.get("seed", 42)),
        eval_fraction=float(data_config.get("eval_fraction", 0.0)),
    )
    eval_fraction = float(data_config.get("eval_fraction", 0.0))
    eval_dataset = None
    if eval_fraction > 0:
        eval_dataset = LunarCLIPDataset(
            captions_path=data_config["captions_path"],
            vision_dataset=vision_dataset,
            modality=modality,
            caption_policy=data_config.get("caption_policy", "sample_one"),
            split="eval",
            seed=int(training_config.get("seed", 42)),
            eval_fraction=eval_fraction,
        )
    clip_model = LunarCLIPModel(
        text_adapter=text_adapter,
        vision_adapter=vision_adapter,
        projection_dim=int(model_config.get("projection_dim", 512)),
        temperature=float(model_config.get("temperature", 0.07)),
    )
    clip_training_config = LunarCLIPTrainingConfig(
        output_dir=training_config.get("output_dir", "results/clip"),
        run_name=training_config.get("run_name", experiment_config["name"]),
        batch_size=int(training_config.get("batch_size", 32)),
        epochs=int(training_config.get("epochs", 1)),
        learning_rate=float(training_config.get("learning_rate", 1e-4)),
        text_encoder_learning_rate=float(
            training_config.get(
                "text_encoder_learning_rate",
                training_config.get("learning_rate", 1e-4),
            )
        ),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        num_workers=int(training_config.get("num_workers", 0)),
        seed=int(training_config.get("seed", 42)),
        accelerator=training_config.get("accelerator", "auto"),
        devices=training_config.get("devices", "auto"),
        precision=training_config.get("precision", "32-true"),
        modality=modality,
        grad_clip=float(training_config.get("grad_clip", 1.0)),
        log_every=int(training_config.get("log_every", 50)),
        save_top_k=int(training_config.get("save_top_k", 1)),
    )
    return clip_model, train_dataset, eval_dataset, clip_training_config


def _build_retrieval_evaluation_metadata(config):
    from lunar_clip.data.schemas import MULTI_CAPTION_SOURCES
    from lunar_clip.retrieval import RetrievalEvaluationMetadata

    experiment_config = config["experiment"]
    text_config = config["text"]
    vision_config = config["vision"]
    data_config = config["data"]
    model_config = config.get("model", {})
    training_config = config["training"]
    caption_policy = data_config.get("caption_policy", "sample_one")
    caption_sources = [
        {"source_version": source_version, "prompt_style": prompt_style}
        for source_version, prompt_style in MULTI_CAPTION_SOURCES.get(
            caption_policy, ()
        )
    ]
    return RetrievalEvaluationMetadata(
        model={
            "text_encoder": text_config["encoder"],
            "vision_encoder": vision_config["encoder"],
            "modality": experiment_config["modality"],
            "projection_dim": int(model_config.get("projection_dim", 512)),
            "temperature": float(model_config.get("temperature", 0.07)),
        },
        dataset={
            "captions_path": data_config["captions_path"],
            "vision_root": data_config["vision_root"],
            "patch_size": int(data_config["patch_size"]),
            "stride": int(data_config["stride"]),
            "caption_policy": caption_policy,
            "caption_sources": caption_sources,
            "eval_fraction": float(data_config.get("eval_fraction", 0.0)),
            "batch_size": int(training_config.get("batch_size", 32)),
        },
        training={
            "run_name": training_config.get("run_name", experiment_config["name"]),
            "seed": int(training_config.get("seed", 42)),
            "epochs": int(training_config.get("epochs", 1)),
            "learning_rate": float(training_config.get("learning_rate", 1e-4)),
            "text_encoder_learning_rate": float(
                training_config.get(
                    "text_encoder_learning_rate",
                    training_config.get("learning_rate", 1e-4),
                )
            ),
            "weight_decay": float(training_config.get("weight_decay", 0.01)),
            "text_encoder_frozen": bool(text_config.get("freeze_encoder", False)),
            "vision_encoder_frozen": bool(vision_config.get("freeze_encoder", False)),
        },
    )


def _vision_transforms(modality):
    if modality != "wac":
        return None
    return {"wac": _normalize_wac}


def _normalize_wac(tensor):
    return (tensor - WAC_NORMALIZE_MEAN) / WAC_NORMALIZE_STD
