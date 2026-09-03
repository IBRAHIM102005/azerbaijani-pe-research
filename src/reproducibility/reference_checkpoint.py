"""SYNTHETIC_FIXTURE reference checkpoint API.

This is NOT Fidan's checkpoint system. It exists only so
tests/integration/test_checkpoint_integration.py can exercise the
save/load/resume-equivalence *test pattern* end-to-end before Fidan's real
`training.checkpoint` module exists.

Per the interface contract (docs/INTERFACE_CONTRACT.md), Fidan's real module
must expose the same two functions with the same signatures; once it does,
point `src/reproducibility/adapters.py::checkpoint_adapter()` at it and these tests will
exercise the real thing with zero changes to the test bodies.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: dict,
    tokens_seen: int,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": rng_state,
            "tokens_seen": tokens_seen,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "tokens_seen": ckpt["tokens_seen"],
        "rng_state": ckpt["rng_state"],
        "extra": ckpt.get("extra", {}),
    }


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
