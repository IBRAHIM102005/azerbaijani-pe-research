"""Optimizer and learning-rate schedule for M3 training."""

from __future__ import annotations

import math

import torch


def build_optimizer(
    model: torch.nn.Module,
    *,
    peak_lr: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Build the shared AdamW optimizer used by every PE run."""

    kwargs = {
        "lr": peak_lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
    }

    # Fused AdamW can improve CUDA throughput.
    # We keep it optional because it may not be supported
    # on every PyTorch/device combination.
    if fused is not None:
        kwargs["fused"] = fused

    return torch.optim.AdamW(
        model.parameters(),
        **kwargs,
    )


def learning_rate_at_step(
    step: int,
    total_steps: int,
    *,
    peak_lr: float = 6e-4,
    warmup_ratio: float = 0.01,
    min_lr_ratio: float = 0.10,
) -> float:
    """Linear warmup followed by cosine decay.

    The schedule is defined directly as a function of optimizer step.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if step < 0:
        raise ValueError("step cannot be negative")

    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr = peak_lr * min_lr_ratio

    # Linear warmup:
    # 0 -> peak_lr
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Once training is complete, stay at min LR.
    if step >= total_steps:
        return min_lr

    decay_steps = total_steps - warmup_steps

    progress = (step - warmup_steps) / max(1, decay_steps)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + (peak_lr - min_lr) * cosine


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    """Set the same LR for every optimizer parameter group."""

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr