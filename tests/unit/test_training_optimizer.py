import math

import torch

from src.training.optimizer import (
    build_optimizer,
    learning_rate_at_step,
    set_optimizer_lr,
)


def test_build_optimizer_uses_expected_adamw_defaults():
    model = torch.nn.Linear(4, 2)

    optimizer = build_optimizer(model)

    group = optimizer.param_groups[0]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert group["lr"] == 6e-4
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.1


def test_lr_warmup_increases():
    total_steps = 1000

    lr0 = learning_rate_at_step(0, total_steps)
    lr5 = learning_rate_at_step(5, total_steps)
    lr9 = learning_rate_at_step(9, total_steps)

    assert lr0 < lr5 < lr9
    assert math.isclose(lr9, 6e-4)


def test_lr_cosine_decay_approaches_minimum():
    total_steps = 1000

    lr_middle = learning_rate_at_step(
        500,
        total_steps,
    )

    lr_end = learning_rate_at_step(
        total_steps,
        total_steps,
    )

    assert lr_middle > lr_end
    assert math.isclose(
        lr_end,
        6e-5,
        rel_tol=1e-8,
    )


def test_set_optimizer_lr():
    model = torch.nn.Linear(4, 2)
    optimizer = build_optimizer(model)

    set_optimizer_lr(
        optimizer,
        1.23e-4,
    )

    assert optimizer.param_groups[0]["lr"] == 1.23e-4