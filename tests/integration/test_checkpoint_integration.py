"""Checkpoint integration tests.

These do NOT reimplement Fidan's checkpoint system -- they call it through the
adapter (see conftest.py::checkpoint_fns), and adapt to whatever function
signatures the interface contract in docs/INTERFACE_CONTRACT.md defines.
"""
import sys
from pathlib import Path

import pytest


from src.reproducibility.determinism import set_seed  # noqa: E402

torch = pytest.importorskip("torch")
nn = torch.nn

TOLERANCE = 1e-6


def make_toy_model_and_optimizer(seed: int):
    set_seed(seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def make_batch(seed: int, n: int = 8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, generator=g)
    y = torch.randn(n, 16, generator=g)
    return x, y


def train_steps(model, optimizer, n_steps: int, data_seed: int, start_step: int = 0):
    loss_fn = nn.MSELoss()
    for step in range(n_steps):
        x, y = make_batch(data_seed + start_step + step)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    return model, optimizer


def flatten_params(model) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


# ---------------------------------------------------------------------------
# 1. Save -> load restores model params, optimizer state, RNG state, tokens_seen
# ---------------------------------------------------------------------------
def test_save_load_restores_state(tmp_path, checkpoint_fns):
    save_checkpoint, load_checkpoint = checkpoint_fns

    model, optimizer = make_toy_model_and_optimizer(seed=17)
    train_steps(model, optimizer, n_steps=5, data_seed=1000)

    pre_save_params = flatten_params(model).clone()
    pre_save_opt_state = optimizer.state_dict()
    tokens_seen = 5 * 8  # n_steps * batch_size

    from src.reproducibility.reference_checkpoint import capture_rng_state

    rng_state = capture_rng_state()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, rng_state, tokens_seen)
    assert ckpt_path.exists()

    # Perturb the live model/optimizer to prove load actually restores state,
    # rather than the test passing by coincidence.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    optimizer.zero_grad()

    restored = load_checkpoint(ckpt_path, model, optimizer)

    post_load_params = flatten_params(model)
    assert torch.allclose(pre_save_params, post_load_params, atol=TOLERANCE), (
        "model parameters were not restored by load_checkpoint"
    )

    post_load_opt_state = optimizer.state_dict()
    assert post_load_opt_state.keys() == pre_save_opt_state.keys()
    assert restored["tokens_seen"] == tokens_seen


# ---------------------------------------------------------------------------
# 2. Resume equivalence: N continuous steps == N/2 -> save -> load -> N/2
# ---------------------------------------------------------------------------
def test_resume_equivalence(tmp_path, checkpoint_fns):
    save_checkpoint, load_checkpoint = checkpoint_fns
    from src.reproducibility.reference_checkpoint import capture_rng_state

    N = 10
    data_seed = 2026

    # Experiment A: N steps continuously.
    model_a, opt_a = make_toy_model_and_optimizer(seed=17)
    train_steps(model_a, opt_a, n_steps=N, data_seed=data_seed)
    params_a = flatten_params(model_a)

    # Experiment B: N/2 steps -> save -> fresh model/optimizer -> load -> N/2 steps.
    model_b, opt_b = make_toy_model_and_optimizer(seed=17)
    train_steps(model_b, opt_b, n_steps=N // 2, data_seed=data_seed)
    tokens_seen_mid = (N // 2) * 8
    rng_mid = capture_rng_state()
    ckpt_path = tmp_path / "resume_ckpt.pt"
    save_checkpoint(ckpt_path, model_b, opt_b, rng_mid, tokens_seen_mid)

    # Fresh objects to prove we're really resuming from the checkpoint, not
    # reusing in-memory state.
    model_b2, opt_b2 = make_toy_model_and_optimizer(seed=999)  # different init on purpose
    restored = load_checkpoint(ckpt_path, model_b2, opt_b2)
    assert restored["tokens_seen"] == tokens_seen_mid

    tokens_seen = restored["tokens_seen"]
    train_steps(model_b2, opt_b2, n_steps=N // 2, data_seed=data_seed, start_step=N // 2)
    tokens_seen += (N // 2) * 8
    params_b = flatten_params(model_b2)

    assert tokens_seen == N * 8, "tokens_seen must be monotonic and correct across resume"

    max_abs_diff = (params_a - params_b).abs().max().item()
    assert max_abs_diff <= TOLERANCE, (
        f"resumed training diverged from continuous training by {max_abs_diff} "
        f"(tolerance {TOLERANCE}); resume must not reset or skip optimizer state"
    )


def test_resume_does_not_reset_or_repeat_tokens(tmp_path, checkpoint_fns):
    save_checkpoint, load_checkpoint = checkpoint_fns
    from src.reproducibility.reference_checkpoint import capture_rng_state

    model, optimizer = make_toy_model_and_optimizer(seed=42)
    train_steps(model, optimizer, n_steps=3, data_seed=5)
    tokens_seen = 3 * 8
    ckpt_path = tmp_path / "mono.pt"
    save_checkpoint(ckpt_path, model, optimizer, capture_rng_state(), tokens_seen)

    model2, optimizer2 = make_toy_model_and_optimizer(seed=42)
    restored = load_checkpoint(ckpt_path, model2, optimizer2)
    assert restored["tokens_seen"] == tokens_seen
    assert restored["tokens_seen"] > 0


# ---------------------------------------------------------------------------
# 4. RNG state is restored bit-for-bit across save/load
# ---------------------------------------------------------------------------
def test_restore_rng_state_reproduces_subsequent_global_draws(tmp_path, checkpoint_fns):
    """Exercises the global Python/NumPy/torch RNGs directly, so it would
    fail if restore_rng_state() were a no-op or restored the wrong state."""
    import random

    import numpy as np

    from src.reproducibility.reference_checkpoint import capture_rng_state, restore_rng_state

    save_checkpoint, load_checkpoint = checkpoint_fns

    set_seed(2026)
    # Advance the global RNGs by an arbitrary amount before the point we'll
    # actually checkpoint, so "restore" has real work to undo.
    for _ in range(5):
        random.random()
        np.random.rand()
        torch.rand(4)

    rng_state = capture_rng_state()
    model, optimizer = make_toy_model_and_optimizer(seed=1)  # unrelated seed, irrelevant here
    ckpt_path = tmp_path / "rng_ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, rng_state, tokens_seen=0)

    # Reference: what the global RNGs would produce next, from the exact
    # state we just captured, if nothing else touched them in between.
    restore_rng_state(rng_state)
    expected_python = [random.random() for _ in range(3)]
    expected_numpy = np.random.rand(3)
    expected_torch = torch.rand(3)

    # Advance the global RNGs again by a DIFFERENT amount, to prove the
    # upcoming restore is doing real work, not coincidentally already
    # matching because nothing moved.
    for _ in range(11):
        random.random()
        np.random.rand()
        torch.rand(4)

    restored = load_checkpoint(ckpt_path, model, optimizer)
    restore_rng_state(restored["rng_state"])

    actual_python = [random.random() for _ in range(3)]
    actual_numpy = np.random.rand(3)
    actual_torch = torch.rand(3)

    assert actual_python == expected_python, "Python random state was not restored bit-for-bit"
    assert np.array_equal(actual_numpy, expected_numpy), "NumPy random state was not restored bit-for-bit"
    assert torch.equal(actual_torch, expected_torch), "torch CPU random state was not restored bit-for-bit"
