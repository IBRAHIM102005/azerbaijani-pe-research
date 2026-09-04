"""Core training loop utilities for M3."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

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

    # Current unfinished gradient-accumulation cycle.
    accumulated_microbatches: int = 0
    accumulated_loss_tokens: int = 0


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
    """Single-device causal-LM trainer.

    Supports:
        - gradient accumulation
        - token-weighted gradient averaging
        - gradient clipping
        - fp32 / bf16 / fp16 AMP
        - GradScaler for fp16
        - resumable training state
        - final partial accumulation flush
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

        if self.precision == "fp16":
            self.scaler: torch.amp.GradScaler | None = (
                torch.amp.GradScaler("cuda")
            )
        else:
            self.scaler = None

        self.optimizer.zero_grad(
            set_to_none=True
        )

    # --------------------------------------------------
    # Precision
    # --------------------------------------------------

    def _resolve_precision(
        self,
        precision: str,
    ) -> str:

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

        if precision == "auto":

            if self.device.type != "cuda":
                return "fp32"

            if torch.cuda.is_bf16_supported():
                return "bf16"

            return "fp16"

        if precision in {
            "bf16",
            "fp16",
        }:

            if self.device.type != "cuda":
                raise ValueError(
                    f"{precision} training "
                    "currently requires CUDA."
                )

        return precision

    def _autocast_context(self):

        if self.precision == "fp32":
            return nullcontext()

        dtype = (
            torch.bfloat16
            if self.precision == "bf16"
            else torch.float16
        )

        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
        )

    # --------------------------------------------------
    # Learning rate
    # --------------------------------------------------

    def _current_lr(self) -> float:

        return learning_rate_at_step(
            self.state.optimizer_step,
            self.total_steps,
            peak_lr=self.peak_lr,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
        )

    # --------------------------------------------------
    # Checkpoint state
    # --------------------------------------------------

    @property
    def at_accumulation_boundary(self) -> bool:
        """True when no unsaved partial gradients exist."""

        return (
            self.state.accumulated_microbatches
            == 0
        )

    def state_dict(self) -> dict[str, Any]:

        scaler_state = None

        if self.scaler is not None:
            scaler_state = (
                self.scaler.state_dict()
            )

        return {
            "training_state": {
                "micro_step": (
                    self.state.micro_step
                ),
                "optimizer_step": (
                    self.state.optimizer_step
                ),
                "tokens_seen": (
                    self.state.tokens_seen
                ),
                "accumulated_microbatches": (
                    self.state
                    .accumulated_microbatches
                ),
                "accumulated_loss_tokens": (
                    self.state
                    .accumulated_loss_tokens
                ),
            },
            "trainer_config": {
                "total_steps": (
                    self.total_steps
                ),
                "grad_accum_steps": (
                    self.grad_accum_steps
                ),
                "grad_clip": (
                    self.grad_clip
                ),
                "peak_lr": (
                    self.peak_lr
                ),
                "warmup_ratio": (
                    self.warmup_ratio
                ),
                "min_lr_ratio": (
                    self.min_lr_ratio
                ),
                "precision": (
                    self.precision
                ),
            },
            "scaler_state_dict": (
                scaler_state
            ),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:

        if "training_state" not in state:
            raise ValueError(
                "Missing training_state."
            )

        if "trainer_config" not in state:
            raise ValueError(
                "Missing trainer_config."
            )

        saved_config = (
            state["trainer_config"]
        )

        expected_config = {
            "total_steps": (
                self.total_steps
            ),
            "grad_accum_steps": (
                self.grad_accum_steps
            ),
            "grad_clip": (
                self.grad_clip
            ),
            "peak_lr": (
                self.peak_lr
            ),
            "warmup_ratio": (
                self.warmup_ratio
            ),
            "min_lr_ratio": (
                self.min_lr_ratio
            ),
            "precision": (
                self.precision
            ),
        }

        for key, expected in (
            expected_config.items()
        ):

            actual = saved_config.get(
                key
            )

            if actual != expected:
                raise ValueError(
                    "Trainer config mismatch "
                    f"while resuming: {key}: "
                    f"checkpoint={actual!r}, "
                    f"current={expected!r}"
                )

        saved = state[
            "training_state"
        ]

        self.state = TrainingState(
            micro_step=int(
                saved["micro_step"]
            ),
            optimizer_step=int(
                saved["optimizer_step"]
            ),
            tokens_seen=int(
                saved["tokens_seen"]
            ),
            accumulated_microbatches=int(
                saved.get(
                    "accumulated_microbatches",
                    0,
                )
            ),
            accumulated_loss_tokens=int(
                saved.get(
                    "accumulated_loss_tokens",
                    0,
                )
            ),
        )

        # We deliberately save checkpoints only
        # after complete optimizer updates.
        if not self.at_accumulation_boundary:
            raise ValueError(
                "Checkpoint contains an "
                "unfinished accumulation cycle."
            )

        scaler_state = state.get(
            "scaler_state_dict"
        )

        if self.scaler is not None:

            if scaler_state is None:
                raise ValueError(
                    "FP16 resume requires "
                    "GradScaler state."
                )

            self.scaler.load_state_dict(
                scaler_state
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

    # --------------------------------------------------
    # Optimizer update
    # --------------------------------------------------

    def _optimizer_update(
        self,
    ) -> tuple[float, float]:
        """Average accumulated token gradients and update weights."""

        if (
            self.state.accumulated_loss_tokens
            <= 0
        ):
            raise RuntimeError(
                "Cannot optimizer-step with "
                "zero accumulated loss tokens."
            )

        if self.precision == "fp16":

            assert self.scaler is not None

            self.scaler.unscale_(
                self.optimizer
            )

        # Each backward() accumulated the SUM
        # of token losses. Convert that sum into
        # the mean gradient over all valid targets.
        denominator = float(
            self.state
            .accumulated_loss_tokens
        )

        for parameter in (
            self.model.parameters()
        ):

            if parameter.grad is not None:
                parameter.grad.div_(
                    denominator
                )

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
                "Non-finite gradient norm "
                "detected."
            )

        grad_norm = float(
            grad_norm_tensor
            .detach()
            .cpu()
        )

        lr = self._current_lr()

        set_optimizer_lr(
            self.optimizer,
            lr,
        )

        if self.precision == "fp16":

            assert self.scaler is not None

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

        else:

            self.optimizer.step()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.state.optimizer_step += 1

        self.state.accumulated_microbatches = 0
        self.state.accumulated_loss_tokens = 0

        return lr, grad_norm

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train_microbatch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        real_tokens: int | None = None,
    ) -> StepResult:
        """Train on one microbatch.

        ``real_tokens`` is the number of genuine M1
        tokens represented in the batch.

        It differs from ``input_ids.numel()`` only
        for the final padded batch.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence_length]"
            )

        if (
            labels is not None
            and labels.shape
            != input_ids.shape
        ):
            raise ValueError(
                "labels must have the same "
                "shape as input_ids"
            )

        if real_tokens is None:
            real_tokens = (
                input_ids.numel()
            )

        if (
            real_tokens <= 0
            or real_tokens
            > input_ids.numel()
        ):
            raise ValueError(
                "real_tokens must be in "
                "(0, input_ids.numel()]."
            )

        self.model.train()

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        if labels is None:
            labels = input_ids
        else:
            labels = labels.to(
                self.device,
                non_blocking=True,
            )

        # Number of targets that actually
        # contribute to causal-LM CE.
        valid_loss_tokens = int(
            (
                labels[:, 1:]
                != -100
            )
            .sum()
            .item()
        )

        if valid_loss_tokens <= 0:
            raise ValueError(
                "Microbatch contains no valid "
                "next-token training targets."
            )

        # --------------------------
        # Forward
        # --------------------------

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

        if not torch.isfinite(
            loss
        ):

            self.optimizer.zero_grad(
                set_to_none=True
            )

            raise FloatingPointError(
                "Non-finite training loss "
                f"detected: "
                f"{loss.detach().item()}"
            )

        # loss is the MEAN CE over valid
        # targets. Multiplying by the number
        # of targets gives the token-loss sum.
        weighted_loss = (
            loss
            * valid_loss_tokens
        )

        if self.precision == "fp16":

            assert self.scaler is not None

            self.scaler.scale(
                weighted_loss
            ).backward()

        else:

            weighted_loss.backward()

        # --------------------------
        # Counters
        # --------------------------

        self.state.micro_step += 1

        self.state.tokens_seen += int(
            real_tokens
        )

        self.state.accumulated_microbatches += 1

        self.state.accumulated_loss_tokens += (
            valid_loss_tokens
        )

        should_step = (
            self.state
            .accumulated_microbatches
            >= self.grad_accum_steps
        )

        lr = self._current_lr()
        grad_norm = None

        if should_step:

            lr, grad_norm = (
                self._optimizer_update()
            )

        return StepResult(
            loss=float(
                loss.detach().cpu()
            ),
            did_optimizer_step=(
                should_step
            ),
            micro_step=(
                self.state.micro_step
            ),
            optimizer_step=(
                self.state.optimizer_step
            ),
            tokens_seen=(
                self.state.tokens_seen
            ),
            lr=lr,
            grad_norm=grad_norm,
        )

    def finalize_accumulation(
        self,
    ) -> StepResult | None:
        """Flush a final incomplete accumulation cycle.

        Needed when the exact 50M-token stream ends
        before ``grad_accum_steps`` microbatches have
        completed.

        Returns None when no gradients are pending.
        """

        if self.at_accumulation_boundary:
            return None

        lr, grad_norm = (
            self._optimizer_update()
        )

        return StepResult(
            loss=float("nan"),
            did_optimizer_step=True,
            micro_step=(
                self.state.micro_step
            ),
            optimizer_step=(
                self.state.optimizer_step
            ),
            tokens_seen=(
                self.state.tokens_seen
            ),
            lr=lr,
            grad_norm=grad_norm,
        )