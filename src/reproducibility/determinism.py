"""Seed / determinism utilities.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeedReport:
    seed: int
    python_seeded: bool = True
    numpy_seeded: bool = True
    torch_cpu_seeded: bool = False
    torch_cuda_seeded: bool = False
    cuda_available: bool = False
    deterministic_algorithms_requested: bool = False
    deterministic_algorithms_applied: bool = False
    cudnn_deterministic_set: "bool | str" = "unavailable"
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "python_seeded": self.python_seeded,
            "numpy_seeded": self.numpy_seeded,
            "torch_cpu_seeded": self.torch_cpu_seeded,
            "torch_cuda_seeded": self.torch_cuda_seeded,
            "cuda_available": self.cuda_available,
            "deterministic_algorithms_requested": self.deterministic_algorithms_requested,
            "deterministic_algorithms_applied": self.deterministic_algorithms_applied,
            "cudnn_deterministic_set": self.cudnn_deterministic_set,
            "limitations": self.limitations,
        }


KNOWN_LIMITATIONS = (
    "PyTorch does not guarantee bitwise-identical results across different "
    "GPU models, CUDA/cuDNN versions, or hardware/OS platforms, even with "
    "identical seeds and deterministic settings "
    "(see https://docs.pytorch.org/docs/stable/notes/randomness.html). "
    "Only same-image/same-device reproducibility is claimed here.",
)


def set_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """Set Python/NumPy/PyTorch(CPU+CUDA) seeds and record what actually
    happened. Never raises because CUDA/torch happens to be unavailable."""
    report = SeedReport(seed=seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        report.torch_cpu_seeded = True

        report.cuda_available = torch.cuda.is_available()
        if report.cuda_available:
            torch.cuda.manual_seed_all(seed)
            report.torch_cuda_seeded = True

        if deterministic:
            report.deterministic_algorithms_requested = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                report.deterministic_algorithms_applied = True
            except Exception as exc:  # noqa: BLE001
                report.limitations.append(f"use_deterministic_algorithms failed: {exc}")

            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                report.cudnn_deterministic_set = True
            else:
                report.cudnn_deterministic_set = "unavailable"

            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ModuleNotFoundError:
        report.limitations.append("torch not installed in this environment; only Python/NumPy seeded.")

    report.limitations.extend(KNOWN_LIMITATIONS)
    return report
