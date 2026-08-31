"""Reference tests for the five positional encoding schemes.

Each scheme is checked against the defining property from its paper, not
against a stored output of our own code.  A silent bug in RoPE or ALiBi would
otherwise still produce a plausible-looking results table.
"""

import math

import pytest
import torch

from src.models.config import PE_TYPES, ModelConfig
from src.models.positional import (
    AlibiPositionalEncoding,
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
    alibi_slopes,
    build_positional_scheme,
    sinusoidal_table,
)


# ---------------------------------------------------------------------------
# sinusoidal
# ---------------------------------------------------------------------------
def test_sinusoidal_matches_naive_paper_formula():
    """Vaswani et al. (2017) eq. (4)-(5), recomputed elementwise in Python."""
    seq_len, d_model, theta = 17, 32, 10000.0
    table = sinusoidal_table(seq_len, d_model, theta, dtype=torch.float64)

    for pos in range(seq_len):
        for i in range(d_model // 2):
            denom = theta ** (2 * i / d_model)
            assert table[pos, 2 * i].item() == pytest.approx(
                math.sin(pos / denom), abs=1e-9
            )
            assert table[pos, 2 * i + 1].item() == pytest.approx(
                math.cos(pos / denom), abs=1e-9
            )


def test_sinusoidal_rows_are_bounded_and_distinct():
    table = sinusoidal_table(128, 64)
    assert table.abs().max().item() <= 1.0 + 1e-6
    # no two positions share an encoding
    dist = torch.cdist(table, table)
    dist.fill_diagonal_(float("inf"))
    assert dist.min().item() > 1e-3


def test_sinusoidal_has_no_trainable_parameters():
    pe = SinusoidalPositionalEncoding(64, 32)
    assert list(pe.parameters()) == []
    assert pe.is_parametric is False


def test_sinusoidal_extrapolates_beyond_training_length():
    pe = SinusoidalPositionalEncoding(16, 32)
    out = pe.additive_embedding(40, torch.device("cpu"), torch.float32)
    assert out.shape == (1, 40, 32)


# ---------------------------------------------------------------------------
# learned absolute
# ---------------------------------------------------------------------------
def test_learned_table_shape_and_parameter_count():
    pe = LearnedPositionalEncoding(max_seq_len=128, d_model=64)
    assert pe.table.shape == (128, 64)
    assert sum(p.numel() for p in pe.parameters()) == 128 * 64
    assert pe.is_parametric is True


def test_learned_refuses_to_extrapolate():
    pe = LearnedPositionalEncoding(max_seq_len=16, d_model=8)
    with pytest.raises(ValueError, match="does not extrapolate"):
        pe.additive_embedding(17, torch.device("cpu"), torch.float32)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def test_rope_logit_depends_only_on_relative_offset():
    """The defining property: <R(q,m), R(k,n)> is a function of (m - n)."""
    torch.manual_seed(0)
    head_dim, seq_len = 32, 24
    pe = RotaryPositionalEncoding(head_dim, theta=10000.0)

    q = torch.randn(1, 1, seq_len, head_dim, dtype=torch.float64)
    k = torch.randn(1, 1, seq_len, head_dim, dtype=torch.float64)
    pe.inv_freq = pe.inv_freq.double()

    q_rot, k_rot = pe.rotate_qk(q, k)
    scores = (q_rot @ k_rot.transpose(-1, -2))[0, 0]

    # every diagonal of the score matrix corresponds to one offset (m - n)...
    # ...but only if q/k are the *same* vectors shifted, so compare instead the
    # score of the same vector pair placed at two different absolute offsets.
    for shift in (1, 3, 7):
        q_shift, k_shift = pe.rotate_qk(q, k, offset=shift)
        shifted = (q_shift @ k_shift.transpose(-1, -2))[0, 0]
        # position pair (m, n) at offset 0 vs (m+s, n+s) at offset s
        assert torch.allclose(scores, shifted, atol=1e-9), f"offset {shift}"


def test_rope_preserves_vector_norm():
    torch.manual_seed(1)
    pe = RotaryPositionalEncoding(16)
    q = torch.randn(2, 3, 10, 16)
    q_rot, _ = pe.rotate_qk(q, q)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)


def test_rope_is_identity_at_position_zero():
    pe = RotaryPositionalEncoding(8)
    q = torch.randn(1, 1, 1, 8)
    q_rot, _ = pe.rotate_qk(q, q)
    assert torch.allclose(q, q_rot, atol=1e-6)


def test_rope_actually_changes_later_positions():
    """Guards against a no-op RoPE that would silently duplicate the NoPE arm."""
    torch.manual_seed(2)
    pe = RotaryPositionalEncoding(16)
    q = torch.randn(1, 1, 8, 16)
    q_rot, _ = pe.rotate_qk(q, q)
    assert not torch.allclose(q[:, :, 3:], q_rot[:, :, 3:], atol=1e-3)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryPositionalEncoding(15)


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------
def test_alibi_slopes_match_the_paper_for_eight_heads():
    slopes = alibi_slopes(8)
    expected = torch.tensor([2.0 ** (-(i + 1)) for i in range(8)])
    assert torch.allclose(slopes, expected, atol=1e-9)


def test_alibi_slopes_are_geometric_and_decreasing():
    for n_head in (4, 8, 16):
        slopes = alibi_slopes(n_head)
        assert slopes.shape == (n_head,)
        assert torch.all(slopes[:-1] > slopes[1:])
        ratios = slopes[1:] / slopes[:-1]
        assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-6)


def test_alibi_slopes_handle_non_power_of_two_head_counts():
    slopes = alibi_slopes(12)
    assert slopes.shape == (12,)
    assert torch.all(slopes > 0)


def test_alibi_bias_is_zero_on_diagonal_and_linear_in_distance():
    n_head, seq_len = 8, 12
    pe = AlibiPositionalEncoding(n_head)
    bias = pe.attention_bias(seq_len, torch.device("cpu"), torch.float32)[0]

    assert bias.shape == (n_head, seq_len, seq_len)
    diag = torch.diagonal(bias, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))

    slopes = alibi_slopes(n_head)
    for h in range(n_head):
        for i in range(seq_len):
            for j in range(i + 1):
                assert bias[h, i, j].item() == pytest.approx(
                    -slopes[h].item() * (i - j), abs=1e-6
                )


def test_alibi_penalises_distant_keys_more_in_every_head():
    pe = AlibiPositionalEncoding(8)
    bias = pe.attention_bias(16, torch.device("cpu"), torch.float32)[0]
    row = bias[:, 15, :16]                      # last query, all keys
    assert torch.all(row[:, :-1] < row[:, 1:])  # nearer key -> larger (less negative)


# ---------------------------------------------------------------------------
# NoPE
# ---------------------------------------------------------------------------
def test_nope_hooks_are_all_no_ops():
    pe = NoPositionalEncoding()
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    assert pe.additive_embedding(5, torch.device("cpu"), torch.float32) is None
    assert pe.attention_bias(5, torch.device("cpu"), torch.float32) is None
    q_out, k_out = pe.rotate_qk(q, k)
    assert q_out is q and k_out is k
    assert list(pe.parameters()) == []


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_factory_builds_every_arm(pe_type):
    cfg = ModelConfig(pe_type=pe_type, vocab_size=64, d_model=32, n_head=4,
                      n_layer=2, d_ff=64, max_seq_len=16)
    pe = build_positional_scheme(cfg)
    n_params = sum(p.numel() for p in pe.parameters())
    if pe_type == "learned":
        assert n_params == 16 * 32
    else:
        assert n_params == 0


def test_config_rejects_unknown_pe_type():
    with pytest.raises(ValueError, match="unknown pe_type"):
        ModelConfig(pe_type="relative_bias")


# ---------------------------------------------------------------------------
# length-generalisation protocol
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eval_len", [512, 1024, 2048])
def test_length_generalisation_contract(eval_len):
    """Which arms may be evaluated at which context length.

    The evaluation protocol is fixed here so it cannot be decided ad hoc once
    results are in: the four length-agnostic arms are evaluated at every
    context, and the learned arm is reported as n/a beyond 512.
    """
    trained_len = 512
    for pe_type in PE_TYPES:
        cfg = ModelConfig(pe_type=pe_type, max_seq_len=trained_len)
        pe = build_positional_scheme(cfg)
        expected = pe_type != "learned" or eval_len <= trained_len
        assert pe.supports_length(eval_len) is expected, pe_type


def test_learned_arm_refuses_rather_than_interpolating():
    """Silently resizing the table would change the method under study."""
    pe = LearnedPositionalEncoding(max_seq_len=512, d_model=64)
    assert pe.supports_length(512) is True
    assert pe.supports_length(1024) is False
    with pytest.raises(ValueError):
        pe.additive_embedding(1024, torch.device("cpu"), torch.float32)


def test_relative_arms_extrapolate_without_new_parameters():
    for pe_type in ("sinusoidal", "rope", "alibi", "nope"):
        cfg = ModelConfig(pe_type=pe_type, max_seq_len=512)
        pe = build_positional_scheme(cfg)
        assert pe.supports_length(2048) is True
        assert sum(p.numel() for p in pe.parameters()) == 0
