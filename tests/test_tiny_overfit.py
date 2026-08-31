"""Tiny-overfit gate, executed as a test.

The contract is imported from ``src.models.gates`` rather than restated, so
the test and ``scripts/tiny_overfit.py`` cannot disagree about what the gate
is.  A model that cannot memorise 32 short sequences has a bug in its gradient
path, and this must be caught before any GPU time is spent.

Nothing here is an experimental result.
"""

import pytest
import torch

from src.models.config import PE_TYPES
from src.models.gates import (
    GATE_INIT_SEED,
    GATE_LOSS_THRESHOLD,
    GATE_SEQ_LEN,
    GATE_SEQUENCES,
    gate_batch,
    gate_config,
    run_arm,
)
from src.models.transformer import PELanguageModel


# ---------------------------------------------------------------------------
# the preregistered contract itself
# ---------------------------------------------------------------------------
def test_gate_contract_is_the_preregistered_one():
    assert GATE_SEQUENCES == 32
    assert GATE_INIT_SEED in (17, 42, 1234)
    assert GATE_LOSS_THRESHOLD == 0.05


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_gate_uses_zero_dropout_in_every_arm(pe_type):
    assert gate_config(pe_type).dropout == 0.0


def test_gate_batch_is_32_fixed_sequences_identical_every_call():
    first, second = gate_batch(), gate_batch()
    assert first.shape == (GATE_SEQUENCES, GATE_SEQ_LEN) == (32, 64)
    assert torch.equal(first, second)


def test_gate_batch_is_identical_across_arms():
    """Arms must be compared on the same data, even in the gate."""
    batches = [gate_batch() for _ in PE_TYPES]
    for other in batches[1:]:
        assert torch.equal(batches[0], other)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_arm_memorises_the_fixed_batch(pe_type):
    result = run_arm(pe_type)
    assert result.passed, (
        f"{pe_type}: loss only fell from {result.initial_loss:.3f} to "
        f"{result.final_loss:.5f}; threshold is {GATE_LOSS_THRESHOLD}"
    )


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_every_parameter_receives_a_gradient(pe_type):
    """Catches a positional scheme that is registered but never used."""
    model = PELanguageModel(gate_config(pe_type))
    batch = gate_batch()
    _, loss = model(batch, labels=batch)
    loss.backward()

    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0)
    ]
    assert not missing, f"{pe_type}: no gradient reached {missing}"


def test_arms_are_not_secretly_identical():
    """Each active scheme must change the logits relative to NoPE."""
    batch = gate_batch()
    outputs = {}
    for pe_type in PE_TYPES:
        model = PELanguageModel(gate_config(pe_type)).eval()
        with torch.no_grad():
            logits, _ = model(batch)
        outputs[pe_type] = logits

    for pe_type in ("sinusoidal", "rope", "alibi"):
        assert not torch.allclose(outputs[pe_type], outputs["nope"], atol=1e-4), (
            f"{pe_type} is indistinguishable from NoPE at initialisation"
        )


def test_learned_table_actually_trains():
    model = PELanguageModel(gate_config("learned"))
    before = model.pe.table.detach().clone()
    batch = gate_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        _, loss = model(batch, labels=batch)
        loss.backward()
        opt.step()
    assert not torch.allclose(before, model.pe.table.detach(), atol=1e-5)
