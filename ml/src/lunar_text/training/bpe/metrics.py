from __future__ import annotations
import math

import torch


IGNORE_INDEX = -100


def masked_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    predictions = logits.argmax(dim=-1)
    active_positions = labels != ignore_index
    correct = (predictions[active_positions] == labels[active_positions]).sum().item()
    total = active_positions.sum().item()

    return correct / total if total > 0 else 0.0


def masked_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


class RunningMetrics:
    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._total_loss: float = 0.0
        self._total_correct: int = 0
        self._total_masked: int = 0
        self._steps: int = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self._total_loss += loss.item()
        self._steps += 1

        predictions = logits.detach().argmax(dim=-1)
        active = labels != self.ignore_index
        self._total_correct += int((predictions[active] == labels[active]).sum().item())
        self._total_masked += int(active.sum().item())

    @property
    def avg_loss(self) -> float:
        return self._total_loss / self._steps if self._steps > 0 else 0.0

    @property
    def avg_masked_accuracy(self) -> float:
        if self._total_masked == 0:
            return 0.0
        return self._total_correct / self._total_masked

    @property
    def avg_masked_perplexity(self) -> float:
        return masked_perplexity(self.avg_loss)

    @property
    def masked_tokens(self) -> int:
        return self._total_masked

    def summary(self) -> dict[str, float | int]:
        return {
            "loss": round(self.avg_loss, 4),
            "masked_perplexity": round(self.avg_masked_perplexity, 4),
            "masked_accuracy": round(self.avg_masked_accuracy, 4),
            "masked_tokens": self.masked_tokens,
            "steps": self._steps,
        }
