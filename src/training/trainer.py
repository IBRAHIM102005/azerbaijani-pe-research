"""Core training loop utilities for M3."""

from __future__ import annotations

from contextlib import nullcontext
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
    """Single-device trainer with gradient accumulation and AMP.

    Precision modes:
        auto:
            CPU -> fp32
            CUDA with bf16 support -> bf16
            other CUDA -> fp16

        fp32:
            normal full-precision training

        bf16:
            CUDA autocast with bfloat16

        fp16:
            CUDA autocast with float16 + GradScaler
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
        precision: str = "auto",
        state: TrainingState | None = None,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(
                "total_steps must be positive"
            )

        if grad_accum_steps <= 0:
            raise ValueError(
                "grad_accum_steps must be positive"
            )

        if grad_clip <= 0:
            raise ValueError(
                "grad_clip must be positive"
            )

        self.model = model
        self.optimizer = optimizer

        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip

        self.peak_lr = peak_lr
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

        self.device = torch.device(device)

        self.precision = self._resolve_precision(
            precision
        )

        self.state = (
            state
            if state is not None
            else TrainingState()
        )

        self.model.to(self.device)

        # GradScaler is only needed for CUDA fp16.
        if self.precision == "fp16":
            self.scaler: torch.amp.GradScaler | None = (
                torch.amp.GradScaler("cuda")
            )
        else:
            self.scaler = None

        # Start with empty gradient buffers.
        self.optimizer.zero_grad(
            set_to_none=True
        )

    def _resolve_precision(
        self,
        precision: str,
    ) -> str:
        """Resolve requested precision mode."""

        allowed = {
            "auto",
            "fp32",
            "bf16",
            "fp16",
        }

        if precision not in allowed:
            raise ValueError(
                "precision must be one of "
                f"{sorted(allowed)}, "
                f"got {precision!r}"
            )

        # Automatic selection.
        if precision == "auto":
            if self.device.type != "cuda":
                return "fp32"

            if torch.cuda.is_bf16_supported():
                return "bf16"

            return "fp16"

        # Explicit low-precision modes currently
        # require CUDA.
        if precision in {"bf16", "fp16"}:
            if self.device.type != "cuda":
                raise ValueError(
                    f"{precision} training "
                    "currently requires CUDA."
                )

        return precision

    def _autocast_context(self):
        """Return the appropriate autocast context."""

        if self.precision == "fp32":
            return nullcontext()

        if self.precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
        )

    def _current_lr(self) -> float:
        """Compute LR for the current optimizer step."""

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

        Gradients accumulate across microbatches.

        The optimizer updates only after
        ``grad_accum_steps`` microbatches.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence_length]"
            )

        if (
            labels is not None
            and labels.shape != input_ids.shape
        ):
            raise ValueError(
                "labels must have the same "
                "shape as input_ids"
            )

        self.model.train()

        # Transfer batch to CPU/GPU.
        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        if labels is None:
            # PELanguageModel performs the
            # next-token shift internally.
            labels = input_ids
        else:
            labels = labels.to(
                self.device,
                non_blocking=True,
            )

        # -------------------------------
        # Forward pass with AMP
        # -------------------------------

        with self._autocast_context():
            _, loss = self.model(
                input_ids,
                labels=labels,
            )

        if loss is None:
            raise RuntimeError(
                "Model returned loss=None "
                "during training."
            )

        if not torch.isfinite(loss):
            self.optimizer.zero_grad(
                set_to_none=True
            )

            raise FloatingPointError(
                "Non-finite training loss "
                f"detected: {loss.detach().item()}"
            )

        # -------------------------------
        # Gradient accumulation
        # -------------------------------

        scaled_loss = (
            loss / self.grad_accum_steps
        )

        # FP16 needs dynamic loss scaling.
        if self.precision == "fp16":
            assert self.scaler is not None

            self.scaler.scale(
                scaled_loss
            ).backward()

        else:
            scaled_loss.backward()

        # -------------------------------
        # Update bookkeeping
        # -------------------------------

        self.state.micro_step += 1

        self.state.tokens_seen += (
            input_ids.numel()
        )

        should_step = (
            self.state.micro_step
            % self.grad_accum_steps
            == 0
        )

        grad_norm: float | None = None

        # Current LR even if this microbatch
        # does not trigger an optimizer update.
        lr = self._current_lr()

        # -------------------------------
        # Optimizer update
        # -------------------------------

        if should_step:

            # FP16 gradients are currently scaled.
            # Unscale BEFORE gradient clipping.
            if self.precision == "fp16":
                assert self.scaler is not None

                self.scaler.unscale_(
                    self.optimizer
                )

            # Gradient clipping.
            grad_norm_tensor = (
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip,
                )
            )

            if not torch.isfinite(
                grad_norm_tensor
            ):
                self.optimizer.zero_grad(
                    set_to_none=True
                )

                raise FloatingPointError(
                    "Non-finite gradient "
                    "norm detected."
                )

            grad_norm = float(
                grad_norm_tensor
                .detach()
                .cpu()
            )

            # LR changes once per optimizer step,
            # NOT once per microbatch.
            lr = self._current_lr()

            set_optimizer_lr(
                self.optimizer,
                lr,
            )

            # Actual optimizer update.
            if self.precision == "fp16":
                assert self.scaler is not None

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:
                self.optimizer.step()

            # Clear gradients for next
            # accumulation cycle.
            self.optimizer.zero_grad(
                set_to_none=True
            )

            self.state.optimizer_step += 1

        return StepResult(
            loss=float(
                loss.detach().cpu()
            ),
            did_optimizer_step=should_step,
            micro_step=self.state.micro_step,
            optimizer_step=(
                self.state.optimizer_step
            ),
            tokens_seen=self.state.tokens_seen,
            lr=lr,
            grad_norm=grad_norm,
        )