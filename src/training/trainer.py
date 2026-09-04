"""Core training loop utilities for M3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .optimizer import (
    learning_rate_at_step,
    set_optimizer_lr,
)


@dataclass
class TrainingState:
    """Mutable state required to continue a training run."""

    micro_step: int = 0
    optimizer_step: int = 0
    tokens_seen: int = 0


@dataclass
class StepResult:
    """Metrics returned after processing one microbatch."""

    loss: float
    did_optimizer_step: bool
    micro_step: int
    optimizer_step: int
    tokens_seen: int
    lr: float
    grad_norm: float | None


class Trainer:
    """Single-device trainer with gradient accumulation.

    AMP and distributed/parallel execution will be layered on top after
    the basic training semantics are verified.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        peak_lr: float = 6e-4,
        warmup_ratio: float = 0.01,
        min_lr_ratio: float = 0.10,
        device: str | torch.device = "cpu",
        state: TrainingState | None = None,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        if grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

        if grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

        self.model = model
        self.optimizer = optimizer

        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip

        self.peak_lr = peak_lr
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

        self.device = torch.device(device)
        self.state = state or TrainingState()

        self.model.to(self.device)

        # Start with an empty gradient buffer.
        self.optimizer.zero_grad(set_to_none=True)

    def _current_lr(self) -> float:
        return learning_rate_at_step(
            self.state.optimizer_step,
            self.total_steps,
            peak_lr=self.peak_lr,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
        )

    def train_microbatch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StepResult:
        """Train on one microbatch.

        An optimizer update occurs only after ``grad_accum_steps``
        microbatches have accumulated.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence_length]"
            )

        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(
                "labels must have the same shape as input_ids"
            )

        self.model.train()

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        if labels is None:
            # PELanguageModel shifts labels internally.
            labels = input_ids
        else:
            labels = labels.to(
                self.device,
                non_blocking=True,
            )

        _, loss = self.model(
            input_ids,
            labels=labels,
        )

        if loss is None:
            raise RuntimeError(
                "Model returned loss=None during training."
            )

        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)

            raise FloatingPointError(
                f"Non-finite training loss detected: "
                f"{loss.detach().item()}"
            )

        # Gradient accumulation:
        #
        # Instead of using the full global batch at once, split it into
        # several microbatches. Dividing the loss makes the accumulated
        # gradient equivalent to an average across those microbatches.
        scaled_loss = loss / self.grad_accum_steps
        scaled_loss.backward()

        self.state.micro_step += 1
        self.state.tokens_seen += input_ids.numel()

        should_step = (
            self.state.micro_step % self.grad_accum_steps == 0
        )

        grad_norm: float | None = None

        if should_step:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
            )

            grad_norm = float(grad_norm_tensor.detach().cpu())

            if not torch.isfinite(grad_norm_tensor):
                self.optimizer.zero_grad(set_to_none=True)

                raise FloatingPointError(
                    "Non-finite gradient norm detected."
                )

            # LR changes once per optimizer update, not once per
            # microbatch.
            lr = self._current_lr()

            set_optimizer_lr(
                self.optimizer,
                lr,
            )

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.state.optimizer_step += 1

        else:
            # No optimizer update yet.
            lr = self._current_lr()

        return StepResult(
            loss=float(loss.detach().cpu()),
            did_optimizer_step=should_step,
            micro_step=self.state.micro_step,
            optimizer_step=self.state.optimizer_step,
            tokens_seen=self.state.tokens_seen,
            lr=lr,
            grad_norm=grad_norm,
        )