"""Checkpoint save/load utilities for M3 training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy and PyTorch RNG states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore previously captured RNG states safely across devices."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # torch.random.set_rng_state() requires a CPU ByteTensor.
    # A checkpoint loaded with map_location="cuda" may move this
    # state tensor to the GPU, so normalize it back to CPU.
    cpu_rng_state = state["torch_cpu"]

    if not isinstance(cpu_rng_state, torch.Tensor):
        cpu_rng_state = torch.as_tensor(
            cpu_rng_state,
            dtype=torch.uint8,
        )

    cpu_rng_state = (
        cpu_rng_state
        .detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )

    torch.random.set_rng_state(cpu_rng_state)

    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_rng_states = []

        for rng_state in state["torch_cuda"]:
            if not isinstance(rng_state, torch.Tensor):
                rng_state = torch.as_tensor(
                    rng_state,
                    dtype=torch.uint8,
                )

            cuda_rng_states.append(
                rng_state
                .detach()
                .to(device="cpu", dtype=torch.uint8)
                .contiguous()
            )

        torch.cuda.set_rng_state_all(cuda_rng_states)

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: dict,
    tokens_seen: int,
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state,
        "tokens_seen": int(tokens_seen),
        "extra": extra or {},
    }

    # Avoid leaving a half-written checkpoint if training/server dies
    # while torch.save is writing.
    temp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(checkpoint, temp_path)

    os.replace(temp_path, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model/optimizer state and return resume metadata."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "rng_state",
        "tokens_seen",
    }

    missing = required_keys - checkpoint.keys()

    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing required keys: "
            f"{sorted(missing)}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")

        if optimizer_state is None:
            raise ValueError(
                "Optimizer was provided, but checkpoint contains "
                "no optimizer_state_dict."
            )

        optimizer.load_state_dict(optimizer_state)

    return {
        "tokens_seen": int(checkpoint["tokens_seen"]),
        "rng_state": checkpoint["rng_state"],
        "extra": checkpoint.get("extra", {}),
    }