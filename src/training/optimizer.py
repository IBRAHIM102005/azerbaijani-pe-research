"""Optimizer and learning-rate schedule for M3 training."""

from __future__ import annotations

import math
from typing import Any

import torch


# The master plan explicitly requires the learned positional
# table not to be accidentally subjected to weight decay.
#
# In PELanguageModel the parameter is registered as:
#
#     model.pe.table
#
# therefore its named-parameter key is exactly "pe.table".
NO_WEIGHT_DECAY_PARAMETER_NAMES = frozenset(
    {
        "pe.table",
    }
)


def build_optimizer(
    model: torch.nn.Module,
    *,
    peak_lr: float = 6e-4,
    betas: tuple[float, float] = (
        0.9,
        0.95,
    ),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Build the frozen AdamW optimizer.

    Shared model parameters use the configured weight decay.

    The Learned Absolute PE table, when present, is placed into a
    separate zero-weight-decay parameter group. Other PE arms do not
    own this parameter and therefore keep a single optimizer group.
    """

    if peak_lr <= 0:
        raise ValueError(
            "peak_lr must be positive"
        )

    if eps <= 0:
        raise ValueError(
            "eps must be positive"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative"
        )

    decay_params: list[
        torch.nn.Parameter
    ] = []

    no_decay_params: list[
        torch.nn.Parameter
    ] = []

    seen_parameter_ids: set[int] = set()

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        parameter_id = id(
            parameter
        )

        if parameter_id in seen_parameter_ids:
            raise RuntimeError(
                "A trainable parameter appeared "
                f"more than once: {name}"
            )

        seen_parameter_ids.add(
            parameter_id
        )

        if (
            name
            in NO_WEIGHT_DECAY_PARAMETER_NAMES
        ):
            no_decay_params.append(
                parameter
            )

        else:
            decay_params.append(
                parameter
            )

    if not decay_params and not no_decay_params:
        raise ValueError(
            "Model contains no trainable parameters."
        )

    parameter_groups: list[
        dict[str, Any]
    ] = []

    if decay_params:
        parameter_groups.append(
            {
                "params": decay_params,
                "weight_decay": (
                    weight_decay
                ),
            }
        )

    if no_decay_params:
        parameter_groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        )

    kwargs: dict[str, Any] = {
        "lr": peak_lr,
        "betas": betas,
        "eps": eps,

        # Individual groups above explicitly
        # override this when necessary.
        "weight_decay": weight_decay,
    }

    if fused is not None:
        kwargs["fused"] = fused

    return torch.optim.AdamW(
        parameter_groups,
        **kwargs,
    )


def describe_optimizer_groups(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    """Return a human-readable optimizer parameter-group audit."""

    name_by_id = {
        id(parameter): name
        for name, parameter
        in model.named_parameters()
    }

    report: list[
        dict[str, Any]
    ] = []

    for group_index, group in enumerate(
        optimizer.param_groups
    ):
        names: list[str] = []

        num_parameters = 0

        for parameter in group["params"]:
            names.append(
                name_by_id.get(
                    id(parameter),
                    "<unknown>",
                )
            )

            num_parameters += (
                parameter.numel()
            )

        report.append(
            {
                "group_index": (
                    group_index
                ),
                "weight_decay": float(
                    group["weight_decay"]
                ),
                "lr": float(
                    group["lr"]
                ),
                "num_tensors": len(
                    group["params"]
                ),
                "num_parameters": int(
                    num_parameters
                ),
                "parameter_names": names,
            }
        )

    return report


def learning_rate_at_step(
    step: int,
    total_steps: int,
    *,
    peak_lr: float = 6e-4,
    warmup_ratio: float = 0.01,
    min_lr_ratio: float = 0.10,
) -> float:
    """Return LR for one optimizer step.

    Schedule:

        1% linear warmup
              ↓
        peak learning rate
              ↓
        cosine decay
              ↓
        10% of peak LR
    """

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    if step < 0:
        raise ValueError(
            "step cannot be negative"
        )

    if not (
        0.0
        <= warmup_ratio
        < 1.0
    ):
        raise ValueError(
            "warmup_ratio must be in [0, 1)"
        )

    if not (
        0.0
        <= min_lr_ratio
        <= 1.0
    ):
        raise ValueError(
            "min_lr_ratio must be in [0, 1]"
        )

    warmup_steps = max(
        1,
        int(
            total_steps
            * warmup_ratio
        ),
    )

    min_lr = (
        peak_lr
        * min_lr_ratio
    )

    # ------------------------------
    # Linear warmup
    # ------------------------------

    if step < warmup_steps:
        return (
            peak_lr
            * (step + 1)
            / warmup_steps
        )

    # ------------------------------
    # End of schedule
    # ------------------------------

    if step >= total_steps:
        return min_lr

    # ------------------------------
    # Cosine decay
    # ------------------------------

    decay_steps = (
        total_steps
        - warmup_steps
    )

    progress = (
        step
        - warmup_steps
    ) / max(
        1,
        decay_steps,
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi
            * progress
        )
    )

    return (
        min_lr
        + (
            peak_lr
            - min_lr
        )
        * cosine
    )


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    """Set the same LR on every AdamW parameter group."""

    if lr < 0:
        raise ValueError(
            "lr cannot be negative"
        )

    for param_group in (
        optimizer.param_groups
    ):
        param_group["lr"] = lr