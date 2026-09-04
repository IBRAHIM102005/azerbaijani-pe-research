import math

import torch
import torch.nn as nn

from src.training.optimizer import (
    build_optimizer,
    describe_optimizer_groups,
    learning_rate_at_step,
    set_optimizer_lr,
)


def test_build_optimizer_uses_expected_adamw_defaults():
    model = torch.nn.Linear(
        4,
        2,
    )

    optimizer = build_optimizer(
        model
    )

    group = (
        optimizer.param_groups[0]
    )

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )

    assert (
        group["lr"]
        == 6e-4
    )

    assert (
        group["betas"]
        == (
            0.9,
            0.95,
        )
    )

    assert (
        group["eps"]
        == 1e-8
    )

    assert (
        group["weight_decay"]
        == 0.1
    )


def test_lr_warmup_increases():
    total_steps = 1000

    lr0 = learning_rate_at_step(
        0,
        total_steps,
    )

    lr5 = learning_rate_at_step(
        5,
        total_steps,
    )

    lr9 = learning_rate_at_step(
        9,
        total_steps,
    )

    assert (
        lr0
        < lr5
        < lr9
    )

    assert math.isclose(
        lr9,
        6e-4,
    )


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

    assert (
        lr_middle
        > lr_end
    )

    assert math.isclose(
        lr_end,
        6e-5,
        rel_tol=1e-8,
    )


def test_set_optimizer_lr_updates_all_groups():
    model = torch.nn.Linear(
        4,
        2,
    )

    optimizer = build_optimizer(
        model
    )

    set_optimizer_lr(
        optimizer,
        1.23e-4,
    )

    for group in (
        optimizer.param_groups
    ):
        assert (
            group["lr"]
            == 1.23e-4
        )


class ToyLearnedPEModel(
    nn.Module
):
    """Minimal model exposing the same pe.table name as M2."""

    def __init__(self):
        super().__init__()

        self.pe = nn.Module()

        self.pe.table = (
            nn.Parameter(
                torch.randn(
                    8,
                    4,
                )
            )
        )

        self.linear = nn.Linear(
            4,
            4,
        )


def test_learned_positional_table_has_zero_weight_decay():
    model = (
        ToyLearnedPEModel()
    )

    optimizer = build_optimizer(
        model,
        weight_decay=0.1,
    )

    name_by_parameter_id = {
        id(parameter): name
        for name, parameter
        in model.named_parameters()
    }

    decay_by_name = {}

    for group in (
        optimizer.param_groups
    ):
        for parameter in (
            group["params"]
        ):
            name = (
                name_by_parameter_id[
                    id(parameter)
                ]
            )

            decay_by_name[name] = (
                group[
                    "weight_decay"
                ]
            )

    assert (
        decay_by_name[
            "pe.table"
        ]
        == 0.0
    )

    assert (
        decay_by_name[
            "linear.weight"
        ]
        == 0.1
    )

    assert (
        decay_by_name[
            "linear.bias"
        ]
        == 0.1
    )


def test_optimizer_group_audit_reports_parameter_names():
    model = (
        ToyLearnedPEModel()
    )

    optimizer = build_optimizer(
        model,
        weight_decay=0.1,
    )

    report = (
        describe_optimizer_groups(
            model,
            optimizer,
        )
    )

    assert len(report) == 2

    decay_group = next(
        group
        for group in report
        if (
            group["weight_decay"]
            == 0.1
        )
    )

    no_decay_group = next(
        group
        for group in report
        if (
            group["weight_decay"]
            == 0.0
        )
    )

    assert (
        "linear.weight"
        in decay_group[
            "parameter_names"
        ]
    )

    assert (
        "linear.bias"
        in decay_group[
            "parameter_names"
        ]
    )

    assert (
        no_decay_group[
            "parameter_names"
        ]
        == [
            "pe.table",
        ]
    )