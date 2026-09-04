import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.optimizer import (
    build_optimizer,
)
from src.training.trainer import Trainer


class TinyLanguageModel(nn.Module):
    """Cheap model with the same forward contract as PELanguageModel."""

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 8,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.head = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(
        self,
        input_ids,
        labels=None,
    ):
        x = self.embedding(
            input_ids
        )

        logits = self.head(
            x
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(
                    -1,
                    logits.size(-1),
                ),
                labels[:, 1:].reshape(
                    -1
                ),
            )

        return logits, loss


def make_batch():
    return torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )


def test_single_microbatch_updates_model():
    torch.manual_seed(0)

    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        peak_lr=1e-2,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model,
        optimizer,
        total_steps=10,
        grad_accum_steps=1,
        peak_lr=1e-2,
    )

    before = (
        model.head.weight
        .detach()
        .clone()
    )

    result = trainer.train_microbatch(
        make_batch()
    )

    after = (
        model.head.weight
        .detach()
        .clone()
    )

    assert result.did_optimizer_step
    assert result.optimizer_step == 1

    assert not torch.equal(
        before,
        after,
    )


def test_gradient_accumulation_delays_optimizer_step():
    torch.manual_seed(0)

    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        peak_lr=1e-2,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model,
        optimizer,
        total_steps=10,
        grad_accum_steps=2,
        peak_lr=1e-2,
    )

    before = (
        model.head.weight
        .detach()
        .clone()
    )

    first = trainer.train_microbatch(
        make_batch()
    )

    after_first = (
        model.head.weight
        .detach()
        .clone()
    )

    # First microbatch performs backward,
    # but no optimizer update yet.
    assert not first.did_optimizer_step
    assert first.optimizer_step == 0

    assert torch.equal(
        before,
        after_first,
    )

    second = trainer.train_microbatch(
        make_batch()
    )

    after_second = (
        model.head.weight
        .detach()
        .clone()
    )

    # Second microbatch completes
    # the accumulation cycle.
    assert second.did_optimizer_step
    assert second.optimizer_step == 1

    assert not torch.equal(
        before,
        after_second,
    )


def test_tokens_seen_counts_consumed_tokens():
    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model,
        optimizer,
        total_steps=10,
        grad_accum_steps=2,
    )

    batch = make_batch()

    trainer.train_microbatch(
        batch
    )

    result = trainer.train_microbatch(
        batch
    )

    expected = (
        batch.numel() * 2
    )

    assert (
        result.tokens_seen
        == expected
    )


def test_gradients_are_cleared_after_optimizer_step():
    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model,
        optimizer,
        total_steps=10,
        grad_accum_steps=1,
    )

    trainer.train_microbatch(
        make_batch()
    )

    assert all(
        parameter.grad is None
        for parameter
        in model.parameters()
    )


def test_auto_precision_on_cpu_uses_fp32():
    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model,
        optimizer,
        total_steps=10,
        device="cpu",
        precision="auto",
    )

    assert trainer.precision == "fp32"
    assert trainer.scaler is None


def test_explicit_fp16_requires_cuda():
    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        weight_decay=0.0,
    )

    with pytest.raises(
        ValueError,
        match="requires CUDA",
    ):
        Trainer(
            model,
            optimizer,
            total_steps=10,
            device="cpu",
            precision="fp16",
        )


def test_explicit_bf16_requires_cuda():
    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        weight_decay=0.0,
    )

    with pytest.raises(
        ValueError,
        match="requires CUDA",
    ):
        Trainer(
            model,
            optimizer,
            total_steps=10,
            device="cpu",
            precision="bf16",
        )