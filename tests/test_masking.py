"""Causal-masking tests.

A leak from future positions inflates every arm's validation perplexity in the
same direction and would not show up as an obviously broken loss curve, so it
is tested directly and for all five arms.
"""

import pytest
import torch

from src.models.config import PE_TYPES, ModelConfig
from src.models.positional import AlibiPositionalEncoding
from src.models.transformer import PELanguageModel

SEQ_LEN = 16


def tiny_config(pe_type: str) -> ModelConfig:
    return ModelConfig(
        pe_type=pe_type,
        init_seed=17,
        vocab_size=48,
        n_layer=2,
        n_head=4,
        d_model=32,
        d_ff=64,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
    )


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_future_tokens_do_not_change_earlier_logits(pe_type):
    """Perturb position t; every logit at position < t must be unchanged."""
    torch.manual_seed(0)
    model = PELanguageModel(tiny_config(pe_type)).eval()

    ids = torch.randint(0, 48, (1, SEQ_LEN))
    with torch.no_grad():
        base, _ = model(ids)

    t = SEQ_LEN - 4
    perturbed = ids.clone()
    perturbed[0, t] = (perturbed[0, t] + 17) % 48
    assert perturbed[0, t] != ids[0, t]

    with torch.no_grad():
        after, _ = model(perturbed)

    assert torch.allclose(base[:, :t], after[:, :t], atol=1e-5), (
        f"{pe_type}: information leaked backwards from position {t}"
    )
    assert not torch.allclose(base[:, t:], after[:, t:], atol=1e-5), (
        f"{pe_type}: the perturbation had no effect at all -- check the test"
    )


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_prefix_invariance(pe_type):
    """Running a prefix alone gives the same logits as running the full row."""
    torch.manual_seed(1)
    model = PELanguageModel(tiny_config(pe_type)).eval()
    ids = torch.randint(0, 48, (1, SEQ_LEN))
    prefix_len = 9

    with torch.no_grad():
        full, _ = model(ids)
        prefix, _ = model(ids[:, :prefix_len])

    assert torch.allclose(full[:, :prefix_len], prefix, atol=1e-5), (
        f"{pe_type}: prefix logits depend on tokens that follow them"
    )


def test_attention_weights_are_strictly_lower_triangular():
    """Recompute the softmax by hand and check the upper triangle is exactly 0."""
    torch.manual_seed(2)
    model = PELanguageModel(tiny_config("alibi")).eval()
    block = model.blocks[0]
    attn = block.attn

    x = torch.randn(1, SEQ_LEN, 32)
    B, T, C = x.shape
    qkv = attn.qkv(x).view(B, T, 3, attn.n_head, attn.head_dim)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.pe.rotate_qk(q, k)

    mask = attn._causal_mask(T, x.device, q.dtype)
    bias = attn.pe.attention_bias(T, x.device, q.dtype)
    if bias is not None:
        mask = mask + bias

    logits = (q @ k.transpose(-1, -2)) / (attn.head_dim ** 0.5) + mask
    weights = logits.softmax(dim=-1)

    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert weights[..., upper].abs().max().item() == 0.0
    assert torch.allclose(weights.sum(-1), torch.ones(1, attn.n_head, T), atol=1e-5)


def test_alibi_bias_never_rescues_a_masked_position():
    """ALiBi adds a finite bias; -inf from the causal mask must still win."""
    pe = AlibiPositionalEncoding(4)
    bias = pe.attention_bias(8, torch.device("cpu"), torch.float32)
    causal = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
    combined = causal.view(1, 1, 8, 8) + bias
    upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.isinf(combined[..., upper]).all()
    assert torch.isfinite(combined[..., ~upper]).all()


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_loss_is_computed_on_shifted_targets(pe_type):
    """Loss must be next-token prediction, i.e. T-1 supervised positions."""
    torch.manual_seed(3)
    model = PELanguageModel(tiny_config(pe_type)).eval()
    ids = torch.randint(0, 48, (2, SEQ_LEN))
    with torch.no_grad():
        logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, SEQ_LEN, 48)
    assert torch.isfinite(loss)
    # an untrained, uniform-ish model should sit near ln(vocab_size)
    assert 2.0 < loss.item() < 6.0


# ---------------------------------------------------------------------------
# parallel residual (frozen spec)
# ---------------------------------------------------------------------------
def test_block_uses_a_parallel_residual_not_a_sequential_one():
    """The MLP must read the block input, never the attention output.

    Sequential:  x -> x + attn(ln1(x)) -> that + mlp(ln2(x + attn(...)))
    Parallel:    x -> x + attn(ln1(x)) + mlp(ln2(x))          <- required

    The two differ only in what the MLP is fed, so that is what is asserted.
    """
    torch.manual_seed(4)
    model = PELanguageModel(tiny_config("rope")).eval()
    block = model.blocks[0]

    captured = {}
    handle = block.mlp.register_forward_pre_hook(
        lambda _mod, args: captured.__setitem__("mlp_in", args[0].detach().clone())
    )
    x = torch.randn(1, SEQ_LEN, 32)
    with torch.no_grad():
        out = block(x)
    handle.remove()

    expected_parallel = block.ln2(x)
    assert torch.allclose(captured["mlp_in"], expected_parallel, atol=1e-6), (
        "the MLP is not reading the block input -- this is a sequential residual"
    )

    with torch.no_grad():
        attn_out = block.attn(block.ln1(x))
        mlp_out = block.mlp(block.ln2(x))
    assert torch.allclose(out, x + attn_out + mlp_out, atol=1e-6)


def test_parallel_residual_differs_from_sequential():
    """Guards the test above against being vacuously true."""
    torch.manual_seed(5)
    model = PELanguageModel(tiny_config("nope")).eval()
    block = model.blocks[0]
    x = torch.randn(1, SEQ_LEN, 32)

    # at initialisation the residual branches are deliberately small, so
    # amplify the attention branch to make the two layouts clearly distinct
    with torch.no_grad():
        block.attn.proj.weight.mul_(50.0)

    with torch.no_grad():
        parallel = block(x)
        h = x + block.attn(block.ln1(x))
        sequential = h + block.mlp(block.ln2(h))
    assert not torch.allclose(parallel, sequential, atol=1e-4)


# ---------------------------------------------------------------------------
# ALiBi offset contract
# ---------------------------------------------------------------------------
def test_alibi_bias_shape_is_square_for_any_offset():
    pe = AlibiPositionalEncoding(4)
    for offset in (0, 5, 128):
        bias = pe.attention_bias(8, torch.device("cpu"), torch.float32, offset=offset)
        assert bias.shape == (1, 4, 8, 8), f"offset={offset} gave {tuple(bias.shape)}"


def test_alibi_bias_is_offset_invariant_like_rope():
    """ALiBi depends on (m - n) only, so shifting both axes changes nothing."""
    pe = AlibiPositionalEncoding(8)
    base = pe.attention_bias(12, torch.device("cpu"), torch.float32, offset=0)
    for offset in (1, 7, 64):
        shifted = pe.attention_bias(12, torch.device("cpu"), torch.float32, offset=offset)
        assert torch.equal(base, shifted), f"offset={offset} changed the bias"
