from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lunar_text.model.bpe.config import ModelConfig
from lunar_text.model.bpe.model import BPELunarMLM
from lunar_text.training.bpe.dataset import MLMDataset, build_dataset
from lunar_text.training.bpe.logger import MLMRunLogger
from lunar_text.training.bpe.tensorboard_export import log_metrics_to_tensorboard
from lunar_text.utils.checkpoints import checkpoint_value, load_checkpoint
from lunar_text.utils.tokenizers import load_tokenizer


@dataclass
class MLMTrainingConfig:
    train_path: str = "data/mlm/v1.0/train.parquet"
    eval_path: str = "data/mlm/v1.0/eval.parquet"
    tokenizer_path: str = "artifacts/tokenizers/bpe/v4.0/tokenizer.json"
    output_root: str = "results/mlm/runs"
    run_name: str = "mlm_train"
    max_seq_len: int = 384
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    use_lr_scheduler: bool = True
    lr_warmup_steps: int = 0
    lr_warmup_ratio: float = 0.1
    lr_min_ratio: float = 0.0
    max_steps: int = 0
    eval_every: int = 500
    log_every: int = 100
    save_every: int = 1000
    grad_clip: float = 1.0
    device: str = "auto"
    eval_max_batches: int = 50
    num_workers: int = 0
    seed: int = 42
    resume_checkpoint: str | None = None


@dataclass
class MLMTrainingResult:
    run_dir: str
    metrics_path: str
    final_checkpoint_path: str
    global_step: int
    latest_metrics: dict[str, float | int | None]


class BPELunarMLMLightningModule(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: MLMTrainingConfig,
        tokenizer_path: str,
        total_training_steps: int,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer_path = tokenizer_path
        self.total_training_steps = max(1, total_training_steps)
        self.model = BPELunarMLM(model_config)
        self.last_grad_norm: float | None = None
        self.latest_metrics: dict[str, float | int | None] = {}
        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
                "tokenizer_path": tokenizer_path,
            }
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        return self._shared_step(batch, split="train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        return self._shared_step(batch, split="eval")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        if not self.training_config.use_lr_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_linear_warmup_cosine_lambda(
                total_steps=self.total_training_steps,
                warmup_steps=_resolve_warmup_steps(
                    total_steps=self.total_training_steps,
                    warmup_steps=self.training_config.lr_warmup_steps,
                    warmup_ratio=self.training_config.lr_warmup_ratio,
                ),
                min_ratio=self.training_config.lr_min_ratio,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "linear_warmup_cosine",
            },
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        grad_norm = _grad_norm(self.model)
        self.last_grad_norm = grad_norm
        self.log(
            "train_grad_norm",
            torch.tensor(grad_norm, device=self.device),
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["model_config"] = asdict(self.model_config)
        checkpoint["training_config"] = asdict(self.training_config)
        checkpoint["tokenizer_path"] = self.tokenizer_path
        checkpoint["latest_metrics"] = self.latest_metrics
        checkpoint["model_state_dict"] = self.model.state_dict()

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        split: str,
    ) -> dict[str, torch.Tensor]:
        loss, logits = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        masked_correct, masked_tokens = _masked_correct_and_total(
            logits=logits,
            labels=batch["labels"],
        )
        masked_accuracy = masked_correct.float() / masked_tokens.clamp_min(1).float()
        masked_perplexity = torch.exp(loss.detach().clamp(max=80))

        metrics = {
            f"{split}_loss": loss.detach(),
            f"{split}_masked_accuracy": masked_accuracy,
            f"{split}_masked_perplexity": masked_perplexity,
            f"{split}_masked_tokens": masked_tokens.float(),
        }
        output = {
            "loss": loss,
            "masked_correct": masked_correct.detach(),
            "masked_tokens": masked_tokens.detach(),
        }

        if split == "eval":
            levenshtein_total, levenshtein_sequences = _batch_levenshtein_distance(
                logits=logits,
                labels=batch["labels"],
            )
            levenshtein_distance = (
                levenshtein_total.float() / levenshtein_sequences.clamp_min(1).float()
            )
            metrics[f"{split}_levenshtein_distance"] = levenshtein_distance
            output["levenshtein_total"] = levenshtein_total.detach()
            output["levenshtein_sequences"] = levenshtein_sequences.detach()

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=split == "train",
            batch_size=batch["input_ids"].shape[0],
        )

        return output


class MLMDataModule(LightningDataModule):
    def __init__(
        self,
        config: MLMTrainingConfig,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.train_dataset: MLMDataset | None = None
        self.eval_dataset: MLMDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is None:
            self.train_dataset = build_dataset(
                dataset_path=self.config.train_path,
                config=self.model_config,
                tokenizer_path=self.config.tokenizer_path,
            )
        if self.eval_dataset is None:
            self.eval_dataset = build_dataset(
                dataset_path=self.config.eval_path,
                config=self.model_config,
                tokenizer_path=self.config.tokenizer_path,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        if self.train_dataset is None:
            raise RuntimeError("Training dataset was not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=MLMDataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.eval_dataset is None:
            self.setup("validate")
        if self.eval_dataset is None:
            raise RuntimeError("Evaluation dataset was not initialized.")
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=MLMDataset.collate,
        )

    @property
    def train_rows(self) -> int:
        return 0 if self.train_dataset is None else len(self.train_dataset)

    @property
    def eval_rows(self) -> int:
        return 0 if self.eval_dataset is None else len(self.eval_dataset)


class MLMMetricsCSVCallback(Callback):
    def __init__(
        self,
        logger: MLMRunLogger,
        log_every: int,
        start_time: float,
        tensorboard_log_dir: Path,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.log_every = log_every
        self.start_time = start_time
        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        self.train_metrics = _MetricAccumulator()
        self.eval_metrics = _MetricAccumulator()
        self.latest_metrics: dict[str, float | int | None] = {}

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        self.train_metrics.update(outputs)
        step = int(trainer.global_step)
        if step <= 0:
            return

        if step == 1 or step % self.log_every == 0:
            self._write_summary(
                trainer=trainer,
                pl_module=pl_module,
                accumulator=self.train_metrics,
                split="train",
                grad_norm=getattr(pl_module, "last_grad_norm", None),
            )
            self.train_metrics.reset()

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        self.eval_metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        self.eval_metrics.update(outputs)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if trainer.sanity_checking or self.eval_metrics.steps == 0:
            return
        self._write_summary(
            trainer=trainer,
            pl_module=pl_module,
            accumulator=self.eval_metrics,
            split="eval",
            grad_norm=None,
        )

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if self.train_metrics.steps > 0:
            self._write_summary(
                trainer=trainer,
                pl_module=pl_module,
                accumulator=self.train_metrics,
                split="train",
                grad_norm=getattr(pl_module, "last_grad_norm", None),
            )
            self.train_metrics.reset()
        self.writer.flush()
        self.writer.close()

    def _write_summary(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        accumulator: _MetricAccumulator,
        split: str,
        grad_norm: float | None,
    ) -> None:
        summary = accumulator.summary()
        learning_rate = _current_learning_rate(trainer.optimizers)
        row = {
            "elapsed_seconds": round(time.perf_counter() - self.start_time, 3),
            "epoch": int(trainer.current_epoch) + 1,
            "step": int(trainer.global_step),
            "split": split,
            "loss": summary["loss"],
            "masked_accuracy": summary["masked_accuracy"],
            "masked_perplexity": summary["masked_perplexity"],
            "masked_tokens": summary["masked_tokens"],
            "levenshtein_distance": summary["levenshtein_distance"],
            "learning_rate": learning_rate,
            "grad_norm": None if grad_norm is None else round(float(grad_norm), 6),
        }
        self.logger.log_metrics(row)
        log_metrics_to_tensorboard(self.writer, row)
        self.writer.flush()
        self.latest_metrics = summary
        if hasattr(pl_module, "latest_metrics"):
            pl_module.latest_metrics = summary


class _MetricAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_masked = 0
        self.total_levenshtein = 0
        self.levenshtein_sequences = 0
        self.steps = 0

    def update(self, outputs: dict[str, torch.Tensor] | None) -> None:
        if not outputs:
            return

        loss = outputs.get("loss")
        masked_correct = outputs.get("masked_correct")
        masked_tokens = outputs.get("masked_tokens")
        if loss is None or masked_correct is None or masked_tokens is None:
            return

        self.total_loss += float(loss.detach().item())
        self.total_correct += int(masked_correct.detach().item())
        self.total_masked += int(masked_tokens.detach().item())
        self.total_levenshtein += _tensor_int(outputs.get("levenshtein_total"))
        self.levenshtein_sequences += _tensor_int(outputs.get("levenshtein_sequences"))
        self.steps += 1

    def summary(self) -> dict[str, float | int | None]:
        loss = self.total_loss / self.steps if self.steps > 0 else 0.0
        accuracy = (
            self.total_correct / self.total_masked if self.total_masked > 0 else 0.0
        )
        levenshtein_distance = (
            round(self.total_levenshtein / self.levenshtein_sequences, 4)
            if self.levenshtein_sequences > 0
            else None
        )
        return {
            "loss": round(loss, 4),
            "masked_perplexity": round(_masked_perplexity(loss), 4),
            "masked_accuracy": round(accuracy, 4),
            "masked_tokens": self.total_masked,
            "levenshtein_distance": levenshtein_distance,
            "steps": self.steps,
        }


def train_mlm(
    config: MLMTrainingConfig,
    console: Console | None = None,
) -> MLMTrainingResult:
    _validate_training_config(config)
    seed_everything(config.seed, workers=True)

    resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    legacy_checkpoint = _load_legacy_resume_checkpoint(resume_path)
    model_config = (
        _model_config_from_legacy_checkpoint(legacy_checkpoint)
        or _model_config_from_lightning_checkpoint(resume_path)
        or _build_model_config(
            tokenizer_path=config.tokenizer_path,
            max_seq_len=config.max_seq_len,
        )
    )

    data_module = MLMDataModule(config=config, model_config=model_config)
    data_module.setup("fit")
    if data_module.train_rows <= 0:
        raise ValueError("Training split contains no usable rows.")

    run_logger = MLMRunLogger(
        output_root=config.output_root,
        run_name=config.run_name,
    )
    run_logger.save_config(
        {
            "training": asdict(config),
            "model": asdict(model_config),
            "tokenizer_path": config.tokenizer_path,
            "train_rows": data_module.train_rows,
            "eval_rows": data_module.eval_rows,
            "device": config.device,
            "checkpoint_format": "lightning",
            "tensorboard_log_dir": str(run_logger.run_dir),
        }
    )

    lightning_model = BPELunarMLMLightningModule(
        model_config=model_config,
        training_config=config,
        tokenizer_path=config.tokenizer_path,
        total_training_steps=_target_training_steps(
            config, data_module.train_dataloader()
        ),
    )
    if legacy_checkpoint is not None:
        lightning_model.model.load_state_dict(legacy_checkpoint["model_state_dict"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_logger.checkpoints_dir),
        filename="step_{step:06d}",
        every_n_train_steps=config.save_every,
        every_n_epochs=0,
        save_on_train_epoch_end=False,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )
    metrics_callback = MLMMetricsCSVCallback(
        logger=run_logger,
        log_every=config.log_every,
        start_time=time.perf_counter(),
        tensorboard_log_dir=run_logger.run_dir,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(run_logger.run_dir),
        name="",
        version="",
    )

    accelerator, devices = _trainer_accelerator_and_devices(config.device)
    train_batches = len(data_module.train_dataloader())
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.epochs,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, metrics_callback],
        gradient_clip_val=config.grad_clip if config.grad_clip > 0 else None,
        log_every_n_steps=config.log_every,
        val_check_interval=_validation_interval(config, train_batches),
        limit_val_batches=_validation_batch_limit(config, data_module.eval_rows),
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=console is not None,
    )

    if console is not None:
        console.print(
            Panel.fit(
                "[bold blue]Run Initialized[/bold blue]\n"
                f"Run dir    : {run_logger.run_dir}\n"
                f"TensorBoard: {tensorboard_logger.log_dir}\n"
                f"Accelerator: {accelerator}\n"
                f"Train rows : {data_module.train_rows:,}\n"
                f"Eval rows  : {data_module.eval_rows:,}\n"
                f"Vocab size : {model_config.vocab_size:,}"
                + _resume_panel_text(resume_path, legacy_checkpoint),
                title="LunarGeo MLM",
            )
        )

    ckpt_path = (
        str(resume_path) if resume_path and resume_path.suffix == ".ckpt" else None
    )
    trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)

    final_checkpoint_path = run_logger.checkpoints_dir / "final.ckpt"
    lightning_model.latest_metrics = metrics_callback.latest_metrics
    trainer.save_checkpoint(final_checkpoint_path)
    _print_checkpoint(console, final_checkpoint_path)

    return MLMTrainingResult(
        run_dir=str(run_logger.run_dir),
        metrics_path=str(run_logger.metrics_path),
        final_checkpoint_path=str(final_checkpoint_path),
        global_step=int(trainer.global_step),
        latest_metrics=metrics_callback.latest_metrics,
    )


def _build_model_config(tokenizer_path: str, max_seq_len: int) -> ModelConfig:
    tokenizer = load_tokenizer(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        raise ValueError("Tokenizer must contain a [PAD] token.")

    return ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
    )


def _load_legacy_resume_checkpoint(resume_path: Path | None) -> dict[str, Any] | None:
    if resume_path is None:
        return None
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    if resume_path.suffix != ".pt":
        return None

    checkpoint = load_checkpoint(resume_path, map_location="cpu")
    if checkpoint is None:
        raise ValueError(f"Resume checkpoint could not be loaded: {resume_path}")
    required_keys = {"model_state_dict", "model_config", "global_step"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"Legacy resume checkpoint is missing required keys: {sorted(missing_keys)}"
        )
    return checkpoint


def _model_config_from_legacy_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> ModelConfig | None:
    if checkpoint is None:
        return None
    return ModelConfig(**checkpoint["model_config"])


def _model_config_from_lightning_checkpoint(
    resume_path: Path | None,
) -> ModelConfig | None:
    if resume_path is None or resume_path.suffix != ".ckpt":
        return None
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    checkpoint = load_checkpoint(resume_path, map_location="cpu")
    config = checkpoint_value(checkpoint, "model_config")
    return ModelConfig(**config) if config else None


def _trainer_accelerator_and_devices(
    requested_device: str,
) -> tuple[str, int | list[int] | str]:
    if requested_device == "auto":
        return ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
    if requested_device == "cpu":
        return "cpu", 1
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        return "gpu", 1
    if requested_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        device_index = int(requested_device.split(":", 1)[1])
        return "gpu", [device_index]
    return requested_device, 1


def _validation_interval(config: MLMTrainingConfig, train_batches: int) -> int:
    if train_batches <= 0:
        return config.eval_every
    return min(config.eval_every, train_batches)


def _validation_batch_limit(config: MLMTrainingConfig, eval_rows: int) -> int | float:
    if eval_rows <= 0:
        return 0
    return config.eval_max_batches if config.eval_max_batches > 0 else 1.0


def _target_training_steps(config: MLMTrainingConfig, train_loader: DataLoader) -> int:
    epoch_steps = len(train_loader) * config.epochs
    return min(config.max_steps, epoch_steps) if config.max_steps > 0 else epoch_steps


def _validate_training_config(config: MLMTrainingConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if config.epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if config.log_every <= 0:
        raise ValueError("log_every must be greater than 0.")
    if config.eval_every <= 0:
        raise ValueError("eval_every must be greater than 0.")
    if config.save_every <= 0:
        raise ValueError("save_every must be greater than 0.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be greater than or equal to 0.")
    if not 0 <= config.lr_warmup_ratio < 1:
        raise ValueError(
            "lr_warmup_ratio must be greater than or equal to 0 and less than 1."
        )
    if not 0 <= config.lr_min_ratio <= 1:
        raise ValueError("lr_min_ratio must be between 0 and 1.")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be greater than 0.")
    if config.max_steps < 0:
        raise ValueError("max_steps must be greater than or equal to 0.")
    if config.eval_max_batches < 0:
        raise ValueError("eval_max_batches must be greater than or equal to 0.")
    if config.grad_clip < 0:
        raise ValueError("grad_clip must be greater than or equal to 0.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to 0.")
    if config.resume_checkpoint and not str(config.resume_checkpoint).strip():
        raise ValueError("resume_checkpoint cannot be blank.")


def _resolve_warmup_steps(
    total_steps: int,
    warmup_steps: int,
    warmup_ratio: float,
) -> int:
    if warmup_steps > 0:
        return min(warmup_steps, max(0, total_steps - 1))
    return min(int(total_steps * warmup_ratio), max(0, total_steps - 1))


def _linear_warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
):
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(min_ratio, float(current_step + 1) / float(warmup_steps))

        decay_progress = min(
            1.0,
            max(0.0, float(current_step - warmup_steps) / float(decay_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


def _masked_correct_and_total(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.argmax(dim=-1)
    active = labels != ignore_index
    correct = (predictions[active] == labels[active]).sum()
    total = active.sum()
    return correct, total


def _batch_levenshtein_distance(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.detach().argmax(dim=-1).cpu()
    labels_cpu = labels.detach().cpu()
    total_distance = 0
    total_sequences = 0

    for sample_predictions, sample_labels in zip(
        predictions.tolist(),
        labels_cpu.tolist(),
        strict=True,
    ):
        target_sequence: list[int] = []
        prediction_sequence: list[int] = []
        for predicted_token, target_token in zip(
            sample_predictions,
            sample_labels,
            strict=True,
        ):
            if int(target_token) == ignore_index:
                continue
            prediction_sequence.append(int(predicted_token))
            target_sequence.append(int(target_token))

        if not target_sequence:
            continue
        total_distance += _levenshtein_distance(prediction_sequence, target_sequence)
        total_sequences += 1

    return (
        torch.tensor(total_distance, device=logits.device),
        torch.tensor(total_sequences, device=logits.device),
    )


def _levenshtein_distance(left: list[int], right: list[int]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_token != right_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _tensor_int(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return int(value.detach().item())


def _masked_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _grad_norm(model: torch.nn.Module) -> float:
    total_sq_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        total_sq_norm += param_norm * param_norm
    return math.sqrt(total_sq_norm)


def _current_learning_rate(optimizers: list[torch.optim.Optimizer]) -> float:
    if not optimizers:
        return 0.0
    return float(optimizers[0].param_groups[0]["lr"])


def _resume_panel_text(
    resume_path: Path | None,
    legacy_checkpoint: dict[str, Any] | None,
) -> str:
    if resume_path is None:
        return ""
    if resume_path.suffix == ".pt":
        legacy_step = (
            0
            if legacy_checkpoint is None
            else int(legacy_checkpoint.get("global_step") or 0)
        )
        return f"\nLegacy init: {resume_path}\nLegacy step: {legacy_step:,}"
    return f"\nResume    : {resume_path}"


def _print_checkpoint(console: Console | None, checkpoint_path: Path) -> None:
    if console is None:
        return
    console.print(f"[green]Saved checkpoint:[/green] {checkpoint_path}")
