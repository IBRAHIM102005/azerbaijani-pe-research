"""Thin adapters between the reproducibility tooling and other's real modules.

Every function here does ONE of two things:
  1. imports the real other interface and returns it, or
  2. raises a clear, loud error (never silently substitutes a fixture).
"""
from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path
from typing import Callable


class MissingInterfaceError(RuntimeError):
    """Raised when a required other interface is not available."""


def get_model_builder(allow_fixture: bool = False) -> Callable:
    try:
        mod = importlib.import_module("src.models.transformer")
        return mod.build_model
    except ModuleNotFoundError as exc:
        if not allow_fixture:
            raise MissingInterfaceError(
                "other's src.models.transformer.build_model is not importable. "
                "Pass allow_fixture=True only for testing this tooling, "
                "never for a real fairness audit."
            ) from exc
        warnings.warn(
            "other's build_model not found; falling back to the local reference "
            "model (SYNTHETIC_FIXTURE, tests this tooling only, NOT a real "
            "fairness result). Note: the fixture's build_model(pe_type, dict) "
            "signature does NOT match other's real build_model(config) -- callers "
            "must branch on which one they got, or use get_models_module() "
            "instead, which raises rather than silently offering a "
            "mismatched calling convention.",
            stacklevel=2,
        )
        from src.reproducibility.reference_model import build_model as fixture_build_model

        return fixture_build_model


def get_models_module():
    try:
        return importlib.import_module("src.models")
    except ModuleNotFoundError as exc:
        raise MissingInterfaceError(
            "other's src.models package is not importable. Run from the "
            "repository root so 'src' is importable as a top-level package."
        ) from exc


def checkpoint_adapter():
    """Return (save_checkpoint, load_checkpoint) from other's real module.

    Raises MissingInterfaceError if other's module isn't present.
    """
    try:
        mod = importlib.import_module("training.checkpoint")
        return mod.save_checkpoint, mod.load_checkpoint
    except ModuleNotFoundError as exc:
        raise MissingInterfaceError(
            "other's training.checkpoint.{save_checkpoint,load_checkpoint} is "
            "not importable yet."
        ) from exc


def manifest_hash_fn():
    """Return other's sha256_file(path) -> str.

    Confirmed 2026-08-31 by reading other's actual src/data/hashing.py:
    `sha256_file(path: Path, chunk_size: int = 8*1024*1024) -> str`,
    hex digest via hashlib.sha256(...).hexdigest(). The module is
    importable as `src.data.hashing` (other's `src/` has an `__init__.py`,
    so it's a real package rooted at the repo root, not `data.hashing`
    as originally guessed).
    """
    try:
        mod = importlib.import_module("src.data.hashing")
        return mod.sha256_file
    except ModuleNotFoundError as exc:
        raise MissingInterfaceError(
            "other's src.data.hashing.sha256_file is not importable. Run from "
            "the repository root so 'src' is importable as a top-level package."
        ) from exc


def load_training_data_contract(repo_root: str | Path = ".") -> dict:
    path = Path(repo_root) / "data" / "metadata" / "training_data_contract.json"
    if not path.is_file():
        raise MissingInterfaceError(
            f"other's training data contract not found at {path}. Expected "
            "data/metadata/training_data_contract.json (see other's README)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hashes_from_contract(contract: dict) -> dict[str, str]:
    """{"train": sha256, "validation": sha256, "test": sha256} from the contract."""
    manifests = contract.get("artifacts", {}).get("manifests", {})
    missing = sorted({"train", "validation", "test"} - set(manifests))
    if missing:
        raise MissingInterfaceError(
            f"training_data_contract.json is missing manifest entries for {missing}"
        )
    return {split: entry["sha256"] for split, entry in manifests.items()}


def tokenizer_hashes_from_contract(contract: dict) -> dict[str, str]:
    """Per-file tokenizer artifact hashes, e.g. {"tokenizer.model": sha256, ...}."""
    hashes = contract.get("tokenizer", {}).get("artifact_hashes")
    if not hashes:
        raise MissingInterfaceError("training_data_contract.json has no tokenizer.artifact_hashes")
    return hashes


def tokenizer_vocab_size_from_contract(contract: dict) -> int:
    try:
        return contract["tokenizer"]["vocab_size"]
    except KeyError as exc:
        raise MissingInterfaceError("training_data_contract.json missing tokenizer.vocab_size") from exc


def data_seed_from_contract(contract: dict) -> int:
    try:
        return contract["splits"]["seed"]
    except KeyError as exc:
        raise MissingInterfaceError("training_data_contract.json missing splits.seed") from exc
