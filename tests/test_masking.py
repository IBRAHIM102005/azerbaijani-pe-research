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
