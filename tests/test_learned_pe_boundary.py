"""Learned-PE boundary and reference tests.

The learned arm is the only one with trainable positional parameters and the
only one that cannot be evaluated past its trained context.  Both facts need
tests, because both are load-bearing for how the results table is read.
"""

import pytest
import torch

from src.models.config import ModelConfig
from src.models.positional import LearnedPositionalEncoding
from src.models.transformer import PELanguageModel

TRAINED_LEN = 512
SEED = 17


def learned_config(max_seq_len: int = TRAINED_LEN) -> ModelConfig:
    """Small enough to train in a test, with the production context length."""
    return ModelConfig(
        pe_type="learned",
        init_seed=SEED,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        d_model=64,
        d_ff=128,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# the 512 / 513 boundary
# ---------------------------------------------------------------------------
def test_length_512_works():
    model = PELanguageModel(learned_config()).eval()
    ids = torch.randint(0, 64, (1, TRAINED_LEN))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, TRAINED_LEN, 64)


def test_length_513_fails_intentionally():
    """One token past the table must raise, not silently wrap or interpolate."""
    model = PELanguageModel(learned_config()).eval()
    ids = torch.randint(0, 64, (1, TRAINED_LEN + 1))
    with pytest.raises(ValueError, match="does not extrapolate"):
        model(ids)


def test_supports_length_agrees_with_the_runtime_behaviour():
    pe = LearnedPositionalEncoding(TRAINED_LEN, 64)
    assert pe.supports_length(TRAINED_LEN) is True
    assert pe.supports_length(TRAINED_LEN + 1) is False
    pe.additive_embedding(TRAINED_LEN, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError):
        pe.additive_embedding(TRAINED_LEN + 1, torch.device("cpu"), torch.float32)


def test_table_is_not_silently_resized_by_a_long_forward():
    model = PELanguageModel(learned_config())
    before = model.pe.table.shape
    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, TRAINED_LEN + 8)))
    assert model.pe.table.shape == before == (TRAINED_LEN, 64)


# ---------------------------------------------------------------------------
# gradient flow: only rows that were actually used
# ---------------------------------------------------------------------------
def test_only_used_position_rows_receive_gradient():
    """A batch of length L must leave rows L.. untouched by the optimizer."""
    used = 24
    model = PELanguageModel(learned_config())
    ids = torch.randint(0, 64, (2, used))

    _, loss = model(ids, labels=ids)
    loss.backward()

    grad = model.pe.table.grad
    assert grad is not None
    assert grad.shape == (TRAINED_LEN, 64)

    used_norm = grad[:used].abs().sum().item()
    unused_norm = grad[used:].abs().sum().item()
    assert used_norm > 0, "the rows that were used got no gradient"
    assert unused_norm == 0.0, "unused position rows received gradient"


def test_gradient_reaches_every_used_row():
    used = 16
    model = PELanguageModel(learned_config())
    ids = torch.randint(0, 64, (4, used))
    _, loss = model(ids, labels=ids)
    loss.backward()
    per_row = model.pe.table.grad[:used].abs().sum(dim=-1)
    # the final position is never a *source* for a next-token prediction,
    # so rows 0..used-2 must move; row used-1 may legitimately be zero
    assert torch.all(per_row[: used - 1] > 0)


# ---------------------------------------------------------------------------
# trainable and checkpointed
# ---------------------------------------------------------------------------
def test_table_is_a_trainable_parameter():
    model = PELanguageModel(learned_config())
    names = dict(model.named_parameters())
    assert "pe.table" in names
    assert names["pe.table"].requires_grad is True
    assert names["pe.table"].shape == (TRAINED_LEN, 64)


def test_table_moves_under_optimisation():
    model = PELanguageModel(learned_config())
    before = model.pe.table.detach().clone()
    ids = torch.randint(0, 64, (4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        _, loss = model(ids, labels=ids)
        loss.backward()
        opt.step()
    assert not torch.allclose(before, model.pe.table.detach(), atol=1e-5)


def test_table_is_in_the_state_dict_and_survives_a_round_trip(tmp_path):
    model = PELanguageModel(learned_config())
    with torch.no_grad():
        model.pe.table.add_(1.0)          # make it distinguishable
    state = model.state_dict()
    assert "pe.table" in state

    path = tmp_path / "ckpt.pt"
    torch.save(state, path)

    restored = PELanguageModel(learned_config())
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(restored.pe.table, model.pe.table)


def test_other_arms_have_no_positional_entry_in_the_state_dict():
    for pe_type in ("sinusoidal", "rope", "alibi", "nope"):
        cfg = ModelConfig(**{**learned_config().to_dict(), "pe_type": pe_type})
        state = PELanguageModel(cfg).state_dict()
        assert not [k for k in state if k.startswith("pe.")], pe_type
