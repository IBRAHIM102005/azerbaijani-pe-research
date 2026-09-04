import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.batching import (
    SequentialTokenBatcher,
)

from src.training.optimizer import (
    build_optimizer,
)

from src.training.runner import (
    TrainingRunner,
)

from src.training.trainer import (
    Trainer,
)


VOCAB_SIZE = 32
SEQ_LEN = 4
MICRO_BATCH_SEQUENCES = 1
GRAD_ACCUM_STEPS = 2


class TinyLanguageModel(nn.Module):
    """Small model with PELanguageModel-style interface."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
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
                ignore_index=-100,
            )

        return logits, loss


def make_cache(
    path,
    total_tokens,
):
    tokens = (
        np.arange(total_tokens)
        % VOCAB_SIZE
    ).astype(
        np.uint16
    )

    tokens.tofile(
        path
    )


def make_runner(
    path,
    *,
    total_tokens: int,
    seed: int,
) -> TrainingRunner:

    torch.manual_seed(
        seed
    )

    model = TinyLanguageModel()

    optimizer = build_optimizer(
        model,
        peak_lr=1e-2,
        weight_decay=0.0,
    )

    micro_batch_tokens = (
        SEQ_LEN
        * MICRO_BATCH_SEQUENCES
    )

    global_batch_tokens = (
        micro_batch_tokens
        * GRAD_ACCUM_STEPS
    )

    total_steps = math.ceil(
        total_tokens
        / global_batch_tokens
    )

    trainer = Trainer(
        model,
        optimizer,
        total_steps=total_steps,
        grad_accum_steps=(
            GRAD_ACCUM_STEPS
        ),
        peak_lr=1e-2,
        device="cpu",
        precision="fp32",
    )

    batcher = SequentialTokenBatcher(
        path,
        total_tokens=total_tokens,
        seq_len=SEQ_LEN,
        micro_batch_sequences=(
            MICRO_BATCH_SEQUENCES
        ),
        eod_id=1,
    )

    return TrainingRunner(
        trainer,
        batcher,
    )


def test_runner_consumes_exact_token_stream(
    tmp_path,
):
    path = (
        tmp_path
        / "tokens.bin"
    )

    # 18 is deliberately not divisible by
    # the 8-token effective/global batch.
    make_cache(
        path,
        total_tokens=18,
    )

    runner = make_runner(
        path,
        total_tokens=18,
        seed=0,
    )

    summary = runner.run()

    assert summary.start_tokens == 0
    assert summary.end_tokens == 18

    assert (
        runner.trainer.state.tokens_seen
        == 18
    )

    assert (
        runner.batcher.token_offset
        == 18
    )

    assert runner.batcher.exhausted

    # Microbatches:
    #
    # 4 + 4 + 4 + 4 + 2
    #
    # = 5 microbatches.
    assert (
        summary.microbatches_processed
        == 5
    )

    # GAS=2:
    #
    # step after microbatch 2
    # step after microbatch 4
    # final flush after microbatch 5
    assert (
        runner.trainer.state.optimizer_step
        == 3
    )

    assert (
        summary.final_accumulation_flushed
    )

    assert summary.checkpointable


def test_limited_run_does_not_flush_partial_gradients(
    tmp_path,
):
    path = (
        tmp_path
        / "tokens.bin"
    )

    make_cache(
        path,
        total_tokens=24,
    )

    runner = make_runner(
        path,
        total_tokens=24,
        seed=0,
    )

    summary = runner.run(
        max_microbatches=1
    )

    # One microbatch consumed.
    assert summary.end_tokens == 4

    # But GAS=2, so optimizer must not
    # update yet.
    assert (
        runner.trainer.state.optimizer_step
        == 0
    )

    assert not runner.trainer.at_accumulation_boundary

    assert not summary.exhausted

    assert not (
        summary.final_accumulation_flushed
    )

    assert not summary.checkpointable


def test_runner_refuses_checkpoint_mid_accumulation(
    tmp_path,
):
    data_path = (
        tmp_path
        / "tokens.bin"
    )

    checkpoint_path = (
        tmp_path
        / "checkpoint.pt"
    )

    make_cache(
        data_path,
        total_tokens=24,
    )

    runner = make_runner(
        data_path,
        total_tokens=24,
        seed=0,
    )

    runner.run(
        max_microbatches=1
    )

    with pytest.raises(
        RuntimeError,
        match="gradient accumulation",
    ):
        runner.save_checkpoint(
            checkpoint_path
        )


def test_continuous_and_resumed_runner_match(
    tmp_path,
):
    data_path = (
        tmp_path
        / "tokens.bin"
    )

    checkpoint_path = (
        tmp_path
        / "resume.pt"
    )

    total_tokens = 24

    make_cache(
        data_path,
        total_tokens=total_tokens,
    )

    # -----------------------------------
    # Continuous reference run
    # -----------------------------------

    continuous = make_runner(
        data_path,
        total_tokens=total_tokens,
        seed=42,
    )

    continuous.run()

    continuous_params = {
        name: parameter.detach().clone()
        for name, parameter
        in continuous.trainer.model.named_parameters()
    }

    # -----------------------------------
    # Interrupted run
    # -----------------------------------

    interrupted = make_runner(
        data_path,
        total_tokens=total_tokens,
        seed=42,
    )

    # Exactly one accumulation cycle:
    # 2 microbatches × 4 tokens = 8 tokens.
    first_summary = interrupted.run(
        max_microbatches=2
    )

    assert (
        first_summary.end_tokens
        == 8
    )

    assert (
        interrupted.can_checkpoint
    )

    interrupted.save_checkpoint(
        checkpoint_path,
        extra={
            "run_id": (
                "runner-resume-test"
            ),
        },
    )

    # -----------------------------------
    # Simulate a completely new process
    # -----------------------------------

    resumed = make_runner(
        data_path,
        total_tokens=total_tokens,
        seed=9999,
    )

    restored_extra = (
        resumed.load_checkpoint(
            checkpoint_path
        )
    )

    assert (
        restored_extra["run_id"]
        == "runner-resume-test"
    )

    # Exact data position must be restored.
    assert (
        resumed.trainer.state.tokens_seen
        == 8
    )

    assert (
        resumed.batcher.token_offset
        == 8
    )

    # Continue to end.
    resumed.run()

    assert (
        resumed.trainer.state.tokens_seen
        == total_tokens
    )

    assert (
        resumed.batcher.token_offset
        == total_tokens
    )

    assert resumed.batcher.exhausted

    assert (
        resumed.trainer.state.optimizer_step
        == continuous.trainer.state.optimizer_step
    )

    # Most important assertion:
    #
    # uninterrupted training and
    # save -> process dies -> resume
    #
    # end with exactly the same parameters.
    for name, parameter in (
        resumed.trainer.model.named_parameters()
    ):
        assert torch.equal(
            parameter.detach(),
            continuous_params[name],
        )