from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from lunar_clip.data.clip_dataset import LunarCLIPDataset
from lunar_clip.model.lunar_clip_model import LunarCLIPModel, LunarCLIPOutput
from lunar_clip.retrieval.evaluation import (
    RetrievalEvaluationArtifact,
    RetrievalEvaluationMetadata,
    write_retrieval_evaluation_artifact,
)
from lunar_clip.retrieval.metrics import (
    full_index_retrieval_metrics,
    text_to_image_retrieval_metrics,
)
from lunar_clip.training.checkpointing import build_clip_checkpoint_callback
from lunar_clip.training.metrics import in_batch_top1_retrieval_metrics


HIDDEN_TENSORBOARD_TAGS = frozenset({"epoch", "hp_metric"})


@dataclass
class LunarCLIPTrainingConfig:
    output_dir: str = "results/clip"
    run_name: str = "lunar_clip"
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-4
    text_encoder_learning_rate: float | None = None
    weight_decay: float = 0.01
    num_workers: int = 0
    seed: int = 42
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32-true"
    modality: str = "geomap"
    grad_clip: float = 1.0
    log_every: int = 50
    save_top_k: int = 1


@dataclass
class LunarCLIPTrainingResult:
    output_dir: str
    checkpoint_path: str
    global_step: int
    best_model_path: str
    full_retrieval_metrics: dict[str, float]
    evaluation_artifact_paths: list[str]


class CleanCLIPTensorBoardLogger(TensorBoardLogger):
    def log_hyperparams(self, params, metrics=None) -> None:
        del params, metrics

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        visible_metrics = {
            name: value
            for name, value in metrics.items()
            if name not in HIDDEN_TENSORBOARD_TAGS
        }
        if visible_metrics:
            super().log_metrics(visible_metrics, step=step)


class LunarCLIPLightningModule(LightningModule):
    def __init__(
        self,
        model: LunarCLIPModel,
        config: LunarCLIPTrainingConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters({"training_config": asdict(config)})

    def forward(self, text_batch, image_batch, modality: str) -> LunarCLIPOutput:
        return self.model(
            text_batch=text_batch,
            image_batch=image_batch,
            modality=modality,
            return_loss=False,
        )

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        output = self._shared_step(batch=batch, split="train", return_loss=True)
        if output.loss is None:
            raise RuntimeError("LunarCLIPModel did not return a training loss.")
        return output.loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor | None:
        del batch_idx
        output = self._shared_step(batch=batch, split="val", return_loss=True)
        return output.loss

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor | None:
        del batch_idx
        output = self._shared_step(batch=batch, split="test", return_loss=True)
        return output.loss

    def configure_optimizers(self):
        text_parameters = [
            parameter
            for parameter in self.model.text_adapter.parameters()
            if parameter.requires_grad
        ]
        text_parameter_ids = {id(parameter) for parameter in text_parameters}
        other_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in text_parameter_ids
        ]
        parameter_groups: list[dict[str, Any]] = []
        if text_parameters:
            parameter_groups.append(
                {
                    "params": text_parameters,
                    "lr": self.config.text_encoder_learning_rate
                    or self.config.learning_rate,
                }
            )
        if other_parameters:
            parameter_groups.append(
                {"params": other_parameters, "lr": self.config.learning_rate}
            )
        if not parameter_groups:
            raise ValueError("LunarCLIP has no trainable parameters.")
        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.config.weight_decay,
        )

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        train_dataset = getattr(datamodule, "train_dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(int(self.current_epoch))

    def _shared_step(
        self,
        batch: dict[str, Any],
        split: str,
        return_loss: bool,
    ) -> LunarCLIPOutput:
        image_batch = _image_tensor_from_batch(batch, modality=self.config.modality)
        output = self.model(
            text_batch=batch["text"],
            image_batch=image_batch,
            modality=self.config.modality,
            return_loss=return_loss,
            text_patch_ids=batch["text_patch_id"],
            image_patch_ids=batch["image_patch_id"],
        )
        metrics = in_batch_top1_retrieval_metrics(
            output,
            text_patch_ids=batch["text_patch_id"],
            image_patch_ids=batch["image_patch_id"],
        )
        metric_split = "eval" if split == "val" else split
        batch_size = int(batch["image_patch_id"].numel())
        if output.loss is not None:
            if split == "train":
                self._log_metric(
                    "loss_step/train",
                    output.loss,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    batch_size=batch_size,
                )
            self._log_metric(
                f"loss/{metric_split}",
                output.loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        self._log_metrics(
            {f"{name}/{metric_split}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self._log_metric(
            f"logit_scale/{metric_split}",
            output.logit_scale.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return output

    def _log_metric(self, *args, **kwargs) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.log(*args, **kwargs)

    def _log_metrics(self, *args, **kwargs) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.log_dict(*args, **kwargs)


class LunarCLIPDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: LunarCLIPDataset,
        config: LunarCLIPTrainingConfig,
        val_dataset: LunarCLIPDataset | None = None,
        test_dataset: LunarCLIPDataset | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        del dataloader_idx
        return batch

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | list:
        if self.val_dataset is None:
            return []
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader | list:
        if self.test_dataset is None:
            return []
        return self._build_loader(self.test_dataset, shuffle=False)

    def _build_loader(
        self,
        dataset: LunarCLIPDataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=LunarCLIPDataset.collate,
            persistent_workers=self.config.num_workers > 0,
        )


def train_clip(
    model: LunarCLIPModel,
    train_dataset: LunarCLIPDataset,
    config: LunarCLIPTrainingConfig,
    val_dataset: LunarCLIPDataset | None = None,
    test_dataset: LunarCLIPDataset | None = None,
    evaluation_metadata: RetrievalEvaluationMetadata | None = None,
) -> LunarCLIPTrainingResult:
    if evaluation_metadata is None:
        raise ValueError("train_clip requires evaluation_metadata to export retrieval evaluation artifacts.")
    seed_everything(config.seed, workers=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lightning_model = LunarCLIPLightningModule(model=model, config=config)
    datamodule = LunarCLIPDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
    )
    checkpoint_callback = build_clip_checkpoint_callback(
        output_dir=output_dir,
        monitor="loss/eval" if val_dataset is not None else "loss/train",
        save_top_k=config.save_top_k,
    )
    logger = CleanCLIPTensorBoardLogger(
        save_dir=str(output_dir),
        name=config.run_name,
    )
    cuda_context = (
        _temporarily_disable_cuda()
        if str(config.accelerator).lower() == "cpu"
        else nullcontext()
    )
    with cuda_context:
        trainer = Trainer(
            max_epochs=config.epochs,
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            gradient_clip_val=config.grad_clip if config.grad_clip > 0 else None,
            log_every_n_steps=config.log_every,
            callbacks=[checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
        )
        trainer.fit(
            model=lightning_model,
            datamodule=datamodule,
        )
        best_checkpoint_path = _load_best_checkpoint_weights(
            lightning_model=lightning_model,
            checkpoint_path=checkpoint_callback.best_model_path,
        )
        full_retrieval_metrics, evaluation_artifact_paths = _final_full_retrieval_metrics(
            lightning_model=lightning_model,
            datamodule=datamodule,
            checkpoint_path=best_checkpoint_path,
            output_dir=output_dir,
            evaluation_metadata=evaluation_metadata,
        )
        if full_retrieval_metrics:
            logger.log_metrics(full_retrieval_metrics, step=trainer.global_step)
    return LunarCLIPTrainingResult(
        output_dir=str(output_dir),
        checkpoint_path=str(best_checkpoint_path),
        global_step=int(trainer.global_step),
        best_model_path=str(best_checkpoint_path),
        full_retrieval_metrics=full_retrieval_metrics,
        evaluation_artifact_paths=[str(path) for path in evaluation_artifact_paths],
    )


def _load_best_checkpoint_weights(
    lightning_model: LunarCLIPLightningModule,
    checkpoint_path: str,
) -> Path:
    if not checkpoint_path:
        raise RuntimeError(
            "Training completed without a best checkpoint. Set save_top_k to at least 1."
        )
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    lightning_model.load_state_dict(checkpoint["state_dict"])
    return path


def _final_full_retrieval_metrics(
    lightning_model: LunarCLIPLightningModule,
    datamodule: LunarCLIPDataModule,
    checkpoint_path: Path,
    output_dir: Path,
    evaluation_metadata: RetrievalEvaluationMetadata,
) -> tuple[dict[str, float], list[Path]]:
    return evaluate_and_export_retrieval_splits(
        lightning_model=lightning_model,
        loaders={
            "eval": datamodule.val_dataloader(),
            "test": datamodule.test_dataloader(),
        },
        modality=datamodule.config.modality,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        evaluation_metadata=evaluation_metadata,
    )


def evaluate_and_export_retrieval_splits(
    lightning_model: LunarCLIPLightningModule,
    loaders: dict[str, DataLoader | list],
    modality: str,
    checkpoint_path: Path,
    output_dir: Path,
    evaluation_metadata: RetrievalEvaluationMetadata,
) -> tuple[dict[str, float], list[Path]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Cannot export retrieval evaluation results because the checkpoint does not exist: "
            f"{checkpoint_path}"
        )
    metrics: dict[str, float] = {}
    artifact_paths: list[Path] = []
    for split, loader in loaders.items():
        if not isinstance(loader, DataLoader):
            continue
        started_at = perf_counter()
        try:
            split_metrics = evaluate_full_index_retrieval(
                lightning_model=lightning_model,
                dataloader=loader,
                modality=modality,
            )
        except ValueError as exc:
            raise ValueError(
                f"Full-index retrieval evaluation failed for split '{split}': {exc}"
            ) from exc
        latency_seconds = perf_counter() - started_at
        artifact = RetrievalEvaluationArtifact(
            checkpoint={"path": str(checkpoint_path), "name": checkpoint_path.name},
            evaluation={"split": split, "latency_seconds": latency_seconds},
            metrics=split_metrics,
            model=evaluation_metadata.model,
            dataset=evaluation_metadata.dataset,
            training=evaluation_metadata.training,
        )
        artifact_paths.append(
            write_retrieval_evaluation_artifact(artifact=artifact, output_dir=output_dir)
        )
        metrics.update({f"{name}/{split}": value for name, value in split_metrics.items()})
    return metrics, artifact_paths


def evaluate_full_index_retrieval(
    lightning_model: LunarCLIPLightningModule,
    dataloader: DataLoader,
    modality: str,
) -> dict[str, float]:
    model = lightning_model.model
    was_training = model.training
    model.eval()
    device = lightning_model.device

    text_embeds: list[torch.Tensor] = []
    image_embeds: list[torch.Tensor] = []
    text_patch_ids: list[torch.Tensor] = []
    image_patch_ids: list[torch.Tensor] = []
    text_versions: list[str] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device=device)
            image_batch = _image_tensor_from_batch(batch, modality=modality)
            text_embeds.append(model.encode_text(batch["text"]).detach().cpu())
            image_embeds.append(model.encode_image(image_batch, modality=modality).detach().cpu())
            text_patch_ids.append(batch["text_patch_id"].detach().cpu())
            image_patch_ids.append(batch["image_patch_id"].detach().cpu())
            text_versions.extend(batch["text_version"])

    if was_training:
        model.train()
    if not text_embeds:
        raise ValueError("Full-index retrieval evaluation requires at least one batch.")

    all_text_embeds = torch.cat(text_embeds)
    all_image_embeds = torch.cat(image_embeds)
    all_text_patch_ids = torch.cat(text_patch_ids)
    all_image_patch_ids = torch.cat(image_patch_ids)
    if len(text_versions) != all_text_embeds.shape[0]:
        raise ValueError("Full-index evaluation requires one version per text embedding.")

    metrics = full_index_retrieval_metrics(
        text_embeds=all_text_embeds,
        image_embeds=all_image_embeds,
        text_patch_ids=all_text_patch_ids,
        image_patch_ids=all_image_patch_ids,
    )
    for version in sorted(set(text_versions)):
        version_mask = torch.tensor(
            [text_version == version for text_version in text_versions],
            dtype=torch.bool,
        )
        version_metrics = text_to_image_retrieval_metrics(
            text_embeds=all_text_embeds[version_mask],
            image_embeds=all_image_embeds,
            text_patch_ids=all_text_patch_ids[version_mask],
            image_patch_ids=all_image_patch_ids,
        )
        version_slug = _metric_slug(version)
        for name, value in version_metrics.items():
            metrics[name.replace("full_", f"full_{version_slug}_", 1)] = value
    return metrics


def _metric_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device=device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device=device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device=device) for value in batch)
    return batch

# TODO: Move modality-specific batch extraction behind the vision adapter or
# dataset contract. Training and evaluation should not need to know whether a
# backend uses "tensor", "original", or another internal batch key.
def _image_tensor_from_batch(
    batch: dict[str, Any],
    modality: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    try:
        modality_batch = batch["vision"][modality]
    except KeyError as exc:
        raise KeyError(
            f"Batch does not contain vision tensor for modality '{modality}'."
        ) from exc
    if "original" in modality_batch:
        return modality_batch
    image_batch = modality_batch["tensor"]
    if not isinstance(image_batch, torch.Tensor):
        raise TypeError(
            f"Expected vision tensor for modality '{modality}', got {type(image_batch)}."
        )
    return image_batch


class _temporarily_disable_cuda:
    def __enter__(self):
        self._original_is_available = torch.cuda.is_available
        self._original_device_count = torch.cuda.device_count
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        torch.cuda.is_available = self._original_is_available
        torch.cuda.device_count = self._original_device_count
