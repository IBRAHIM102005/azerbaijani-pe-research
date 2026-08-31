"""Reference tests for the five positional encoding schemes.

Each scheme is checked against the defining property from its paper, not
against a stored output of our own code.  A silent bug in RoPE or ALiBi would
otherwise still produce a plausible-looking results table.
"""

import json
import math
from pathlib import Path

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


# ---------------------------------------------------------------------------
# rotary_pct (GPT-NeoX partial rotation)
# ---------------------------------------------------------------------------
def test_rotary_dim_follows_rotary_pct():
    for pct, expected in ((0.25, 8), (0.5, 16), (1.0, 32)):
        pe = RotaryPositionalEncoding(32, rotary_pct=pct)
        assert pe.rotary_dim == expected


def test_partial_rotation_leaves_the_pass_through_channels_untouched():
    torch.manual_seed(10)
    pe = RotaryPositionalEncoding(32, rotary_pct=0.25)
    q = torch.randn(1, 1, 12, 32)
    q_out, _ = pe.rotate_qk(q, q)
    assert torch.allclose(q_out[..., 8:], q[..., 8:], atol=1e-6)
    assert not torch.allclose(q_out[..., :8], q[..., :8], atol=1e-3)


@pytest.mark.parametrize("pct", [0.25, 0.5, 1.0])
def test_relative_offset_invariance_holds_for_any_rotary_pct(pct):
    """The pass-through channels add a position-independent term, so the
    logit still depends only on (m - n)."""
    torch.manual_seed(11)
    pe = RotaryPositionalEncoding(32, rotary_pct=pct)
    pe.inv_freq = pe.inv_freq.double()
    q = torch.randn(1, 1, 16, 32, dtype=torch.float64)
    k = torch.randn(1, 1, 16, 32, dtype=torch.float64)

    base_q, base_k = pe.rotate_qk(q, k)
    base = base_q @ base_k.transpose(-1, -2)
    for shift in (1, 5, 9):
        sq, sk = pe.rotate_qk(q, k, offset=shift)
        assert torch.allclose(base, sq @ sk.transpose(-1, -2), atol=1e-9)


@pytest.mark.parametrize("pct", [0.25, 0.5, 1.0])
def test_partial_rotation_preserves_norm(pct):
    torch.manual_seed(12)
    pe = RotaryPositionalEncoding(32, rotary_pct=pct)
    q = torch.randn(2, 4, 10, 32)
    q_out, _ = pe.rotate_qk(q, q)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)


def test_rotary_pct_values_that_cannot_work_are_rejected():
    with pytest.raises(ValueError, match="rotary_pct must be in"):
        RotaryPositionalEncoding(32, rotary_pct=0.0)
    with pytest.raises(ValueError, match="rotary_pct must be in"):
        RotaryPositionalEncoding(32, rotary_pct=1.5)
    with pytest.raises(ValueError, match="must be even"):
        RotaryPositionalEncoding(32, rotary_pct=0.1)   # -> 3 channels


def test_shipped_rope_config_records_rotary_pct_explicitly():
    """The value must be in the config file, never implicit in the code."""
    path = Path(__file__).resolve().parents[1] / "configs" / "pe" / "rope.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "rotary_pct" in payload
    cfg = ModelConfig.from_dict(payload)
    pe = build_positional_scheme(cfg)
    assert pe.rotary_dim == int(cfg.head_dim * payload["rotary_pct"])


# ---------------------------------------------------------------------------
# rotary_pct matrix
# ---------------------------------------------------------------------------
#: the frozen spec value
SPEC_ROTARY_PCT = 0.25

#: (head_dim, rotary_pct, expected rotated channels)
ROTARY_MATRIX = [
    (32, 0.25, 8),      # the frozen spec geometry: d_model 256 / 8 heads
    (32, 0.50, 16),
    (32, 1.00, 32),
    (16, 0.25, 4),
    (64, 0.25, 16),
    (64, 0.75, 48),
    (128, 0.25, 32),
]

#: (head_dim, rotary_pct, expected error fragment)
INVALID_ROTARY_MATRIX = [
    (32, 0.0, "rotary_pct must be in"),
    (32, -0.25, "rotary_pct must be in"),
    (32, 1.50, "rotary_pct must be in"),
    (32, 0.10, "must be even"),      # -> 3 channels
    (16, 0.10, "must be even"),      # -> 1 channel
    (8, 0.20, "must be even"),       # -> 1 channel
]


def check_rotary_pct(head_dim, rotary_pct, expected_rotary_dim):
    """Assert one (head_dim, rotary_pct) cell behaves as GPT-NeoX specifies.

    Checks the arithmetic, the frequency table size, that exactly the leading
    ``expected_rotary_dim`` channels are rotated, and that the rest are passed
    through untouched.
    """
    pe = RotaryPositionalEncoding(head_dim, rotary_pct=rotary_pct)

    assert pe.rotary_pct == rotary_pct
    assert pe.rotary_dim == expected_rotary_dim == int(head_dim * rotary_pct)
    assert pe.rotary_dim % 2 == 0 and pe.rotary_dim > 0
    assert pe.inv_freq.shape == (expected_rotary_dim // 2,)

    torch.manual_seed(20)
    q = torch.randn(1, 1, 8, head_dim)
    out, _ = pe.rotate_qk(q, q)

    assert out.shape == q.shape
    rotated, passed = out[..., :expected_rotary_dim], out[..., expected_rotary_dim:]
    assert not torch.allclose(rotated, q[..., :expected_rotary_dim], atol=1e-3)
    assert torch.allclose(passed, q[..., expected_rotary_dim:], atol=1e-6)
    assert torch.allclose(q.norm(dim=-1), out.norm(dim=-1), atol=1e-5)
    return pe


@pytest.mark.parametrize("head_dim,rotary_pct,expected", ROTARY_MATRIX)
def test_rotary_pct_matrix(head_dim, rotary_pct, expected):
    check_rotary_pct(head_dim, rotary_pct, expected)


@pytest.mark.parametrize("head_dim,rotary_pct,message", INVALID_ROTARY_MATRIX)
def test_invalid_rotary_pct_matrix(head_dim, rotary_pct, message):
    with pytest.raises(ValueError, match=message):
        RotaryPositionalEncoding(head_dim, rotary_pct=rotary_pct)


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_every_shipped_config_pins_rotary_pct_to_the_spec(pe_type):
    """0.25 must be written in every config file, not defaulted in code."""
    path = Path(__file__).resolve().parents[1] / "configs" / "pe" / f"{pe_type}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["rotary_pct"] == SPEC_ROTARY_PCT, (
        f"{pe_type}.json has rotary_pct={payload.get('rotary_pct')}, "
        f"expected {SPEC_ROTARY_PCT}"
    )


def test_base_config_pins_rotary_pct_to_the_spec():
    path = Path(__file__).resolve().parents[1] / "configs" / "model_base.json"
    assert json.loads(path.read_text(encoding="utf-8"))["rotary_pct"] == SPEC_ROTARY_PCT


def test_spec_rotary_pct_applied_to_the_real_geometry():
    """End to end: shipped rope config -> built scheme -> 8 rotated channels."""
    path = Path(__file__).resolve().parents[1] / "configs" / "pe" / "rope.json"
    cfg = ModelConfig.from_json(path)
    assert cfg.rotary_pct == SPEC_ROTARY_PCT
    assert cfg.head_dim == 32
    pe = build_positional_scheme(cfg)
    check_rotary_pct(cfg.head_dim, SPEC_ROTARY_PCT, 8)
    assert pe.rotary_dim == 8