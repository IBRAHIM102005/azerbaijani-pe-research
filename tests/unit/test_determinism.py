import sys
from pathlib import Path

import pytest


from src.reproducibility.determinism import set_seed  

torch = pytest.importorskip("torch")

TOLERANCE = 0.0  # same-image/same-device CPU op: expect exact bitwise match


def _small_deterministic_op(seed: int) -> "torch.Tensor":
    report = set_seed(seed)
    assert report.torch_cpu_seeded
    x = torch.randn(64, 64)
    w = torch.randn(64, 64)
    y = torch.nn.functional.gelu(x @ w)
    return y


def test_same_seed_same_environment_reproducible():
    out1 = _small_deterministic_op(17)
    out2 = _small_deterministic_op(17)
    max_abs_diff = (out1 - out2).abs().max().item()
    assert max_abs_diff <= TOLERANCE, (
        f"same-seed, same-environment run diverged by {max_abs_diff} "
        f"(tolerance {TOLERANCE})"
    )


def test_different_seed_produces_different_output():
    out1 = _small_deterministic_op(17)
    out2 = _small_deterministic_op(42)
    assert not torch.allclose(out1, out2)


def test_seed_report_records_limitations_honestly():
    report = set_seed(1234)
    assert report.seed == 1234
    assert report.python_seeded and report.numpy_seeded
    assert any("bitwise-identical" in msg for msg in report.limitations)


def test_set_seed_never_raises_without_cuda():
    # cuda_available may be False in CI; the call must still succeed and
    # record that honestly rather than crashing.
    report = set_seed(7)
    if not report.cuda_available:
        assert report.torch_cuda_seeded is False
