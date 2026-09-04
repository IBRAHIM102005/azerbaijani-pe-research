import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.optimizer import (
    build_optimizer,
)

from src.training.resume import (
    load_training_checkpoint,
    save_training_checkpoint,
)

from src.training.trainer import (
    Trainer,
)


class TinyLanguageModel(nn.Module):
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


def make_trainer(seed: int):
    torch.manual_seed(seed)

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
        device="cpu",
        precision="fp32",
    )

    return trainer


def test_checkpoint_requires_accumulation_boundary(
    tmp_path,
):
    trainer = make_trainer(
        seed=0
    )

    trainer.train_microbatch(
        make_batch()
    )

    assert not trainer.at_accumulation_boundary

    path = tmp_path / "bad.pt"

    with pytest.raises(
        RuntimeError,
        match="accumulation boundaries",
    ):
        save_training_checkpoint(
            path,
            trainer,
        )


def test_resume_restores_training_state(
    tmp_path,
):
    trainer = make_trainer(
        seed=0
    )

    batch = make_batch()

    # Two microbatches = one optimizer step.
    trainer.train_microbatch(batch)
    trainer.train_microbatch(batch)

    path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        path,
        trainer,
        extra={
            "run_id": "unit-test",
        },
    )

    fresh_trainer = make_trainer(
        seed=999
    )

    restored = load_training_checkpoint(
        path,
        fresh_trainer,
    )

    assert restored["micro_step"] == 2
    assert restored["optimizer_step"] == 1

    assert restored["tokens_seen"] == (
        batch.numel() * 2
    )

    assert (
        restored["extra"]["run_id"]
        == "unit-test"
    )


def test_resume_restores_model_parameters(
    tmp_path,
):
    trainer = make_trainer(
        seed=0
    )

    batch = make_batch()

    trainer.train_microbatch(batch)
    trainer.train_microbatch(batch)

    expected = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        path,
        trainer,
    )

    fresh_trainer = make_trainer(
        seed=1234
    )

    load_training_checkpoint(
        path,
        fresh_trainer,
    )

    for name, parameter in (
        fresh_trainer.model.named_parameters()
    ):
        assert torch.equal(
            parameter.detach(),
            expected[name],
        )


def test_continuous_and_resumed_training_match(
    tmp_path,
):
    batch = make_batch()

    # ---------------------------------
    # Continuous training
    # ---------------------------------

    continuous = make_trainer(
        seed=42
    )

    for _ in range(4):
        continuous.train_microbatch(
            batch
        )

    continuous_params = {
        name: parameter.detach().clone()
        for name, parameter
        in continuous.model.named_parameters()
    }

    # ---------------------------------
    # Interrupted training
    # ---------------------------------

    interrupted = make_trainer(
        seed=42
    )

    # First accumulation cycle.
    interrupted.train_microbatch(
        batch
    )
    interrupted.train_microbatch(
        batch
    )

    path = tmp_path / "resume.pt"

    save_training_checkpoint(
        path,
        interrupted,
    )

    # Pretend the process/server died.
    resumed = make_trainer(
        seed=9999
    )

    load_training_checkpoint(
        path,
        resumed,
    )

    # Continue remaining microbatches.
    resumed.train_microbatch(
        batch
    )
    resumed.train_microbatch(
        batch
    )

    assert resumed.state.micro_step == 4
    assert resumed.state.optimizer_step == 2

    for name, parameter in (
        resumed.model.named_parameters()
    ):
        assert torch.equal(
            parameter.detach(),
            continuous_params[name],
        )


def test_rng_state_is_restored(
    tmp_path,
):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    trainer = make_trainer(
        seed=123
    )

    path = tmp_path / "rng.pt"

    save_training_checkpoint(
        path,
        trainer,
    )

    # Values that should appear immediately
    # after the saved RNG state.
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    # Perturb all RNGs.
    for _ in range(20):
        random.random()
        np.random.random()
        torch.rand(1)

    load_training_checkpoint(
        path,
        trainer,
    )

    actual_python = random.random()
    actual_numpy = np.random.random()
    actual_torch = torch.rand(1)

    assert actual_python == expected_python

    assert actual_numpy == expected_numpy

    assert torch.equal(
        actual_torch,
        expected_torch,
    )