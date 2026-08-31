"""Tiny-overfit test.

A model that cannot memorise a handful of short sequences has a bug somewhere
in the gradient path -- a detached tensor, a mask applied to the wrong axis, a
positional signal that destroys the residual stream.  Every arm must pass this
before any GPU time is spent on the real 50M-token runs.

The thresholds are deliberately loose: this is a smoke test for wiring, not a
measurement.  Nothing here is reported as a result.
"""

import pytest
import torch

from src.models.config import PE_TYPES, ModelConfig
from src.models.transformer import PELanguageModel

VOCAB = 32
SEQ_LEN = 24
BATCH = 4
STEPS = 220
LOSS_THRESHOLD = 0.15


def overfit_config(pe_type: str) -> ModelConfig:
    return ModelConfig(
        pe_type=pe_type,
        vocab_size=VOCAB,
        n_layer=2,
        n_head=4,
        d_model=64,
        d_ff=128,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        init_seed=2026,
    )


def fixed_batch() -> torch.Tensor:
    gen = torch.Generator().manual_seed(1234)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN), generator=gen)


def run_overfit(pe_type: str, steps: int = STEPS):
    torch.manual_seed(0)
    model = PELanguageModel(overfit_config(pe_type))
    batch = fixed_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    first_loss = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(batch, labels=batch)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
    return first_loss, loss.item(), model


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_arm_can_memorise_a_tiny_batch(pe_type):
    first, final, _ = run_overfit(pe_type)
    assert final < LOSS_THRESHOLD, (
        f"{pe_type}: loss only fell from {first:.3f} to {final:.3f}; "
        "the arm cannot memorise 4 sequences, so something is wired wrong"
    )


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_every_parameter_receives_a_gradient(pe_type):
    """Catches a positional scheme that is registered but never used."""
    model = PELanguageModel(overfit_config(pe_type))
    batch = fixed_batch()
    _, loss = model(batch, labels=batch)
    loss.backward()

    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0)
    ]
    assert not missing, f"{pe_type}: no gradient reached {missing}"


def test_learned_table_actually_trains():
    """The learned arm's positional table must move away from its init."""
    model = PELanguageModel(overfit_config("learned"))
    before = model.pe.table.detach().clone()
    batch = fixed_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        _, loss = model(batch, labels=batch)
        loss.backward()
        opt.step()
    assert not torch.allclose(before, model.pe.table.detach(), atol=1e-5)


def test_arms_are_not_secretly_identical():
    """NoPE and each positional arm must produce different logits at init.

    If a scheme were silently inactive, its arm would be a duplicate of NoPE
    and the study would compare five copies of the same model.
    """
    torch.manual_seed(0)
    batch = fixed_batch()
    outputs = {}
    for pe_type in PE_TYPES:
        model = PELanguageModel(overfit_config(pe_type)).eval()
        with torch.no_grad():
            logits, _ = model(batch)
        outputs[pe_type] = logits

    for pe_type in PE_TYPES:
        if pe_type == "nope":
            continue
        if pe_type == "learned":
            # a zero-mean table barely moves logits at init; skip the check and
            # rely on test_learned_table_actually_trains instead
            continue
        assert not torch.allclose(outputs[pe_type], outputs["nope"], atol=1e-4), (
            f"{pe_type} is indistinguishable from NoPE at initialisation"
        )
