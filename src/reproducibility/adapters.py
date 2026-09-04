"""Thin adapters between the reproducibility tooling and Ibrahim, Yasin, Fidan, and Nihat's real modules.

Every function here does ONE of two things:
  1. imports the real Ibrahim/Yasin/Fidan/Nihat interface and returns it, or
  2. raises a clear, loud error (never silently substitutes a fixture).

The single exception is `get_model_builder(allow_fixture=...)`, which may
optionally fall back to the reference model in `reference_model` -- and
even then it prints an unmistakable warning, because that fallback exists
only so this suite's own tests can run before Yasin lands, per project rule
"Do NOT silently ignore missing dependencies or failed checks."

Edit the `# TODO(integration)` import lines below once Ibrahim/Yasin/Fidan/Nihat's real
modules exist in this repository; nothing else in this test suite should
need to change.
"""
from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path
from typing import Callable


class MissingInterfaceError(RuntimeError):
    """Raised when a required Ibrahim/Yasin/Fidan/Nihat interface is not available."""


def get_model_builder(allow_fixture: bool = False) -> Callable:
    """Return src.models.transformer.build_model.

    Confirmed 2026-08-31 against Yasin's real code: the signature is
    `build_model(config: ModelConfig) -> PELanguageModel` — a single
    resolved ModelConfig object, not `(pe_type, config_dict)` as an earlier
    draft of this adapter guessed. Prefer `get_models_module()` below for
    new code; this function is kept for the `allow_fixture` fallback path
    used only to test this suite's own tooling when Yasin isn't present at all.
    """
    try:
        mod = importlib.import_module("src.models.transformer")
        return mod.build_model
    except ModuleNotFoundError as exc:
        if not allow_fixture:
            raise MissingInterfaceError(
                "Yasin's src.models.transformer.build_model is not importable. "
                "Pass allow_fixture=True only for testing this tooling, "
                "never for a real fairness audit."
            ) from exc
        warnings.warn(
            "Yasin's build_model not found; falling back to the local reference "
            "model (SYNTHETIC_FIXTURE, tests this tooling only, NOT a real "
            "fairness result). Note: the fixture's build_model(pe_type, dict) "
            "signature does NOT match Yasin's real build_model(config) -- callers "
            "must branch on which one they got, or use get_models_module() "
            "instead, which raises rather than silently offering a "
            "mismatched calling convention.",
            stacklevel=2,
        )
        from src.reproducibility.reference_model import build_model as fixture_build_model

        return fixture_build_model


def get_models_module():
    """Return Yasin's real `src.models` package (ModelConfig, PE_TYPES,
    ARM_ALLOWLIST, fairness_report, format_fairness_table, build_model,
    resolve_run_config, iter_runs, load_run_matrix, config_sha256, ...).

    Confirmed 2026-08-31 by reading Yasin's actual code. This suite does not
    reimplement config-drift or parameter-fairness logic: Yasin already
    ships `ARM_ALLOWLIST`-based config-contract enforcement
    (`tests/test_config_contract.py`) and a parameter/init fairness
    report (`src/models/params.py::fairness_report`, more rigorous than
    an earlier placeholder here -- it also checks bit-identical shared-weight
    initialization across arms). This suite's job is to run these in CI/Makefile
    with clear exit codes and JSON artifacts, not to re-derive what counts
    as "fair". No fixture fallback here: unlike parameter counting, there
    is no honest way to "test the tooling" against a fake config-contract
    module, since the contract IS Yasin's module.
    """
    try:
        return importlib.import_module("src.models")
    except ModuleNotFoundError as exc:
        raise MissingInterfaceError(
            "Yasin's src.models package is not importable. Run from the "
            "repository root so 'src' is importable as a top-level package."
        ) from exc


def checkpoint_adapter():
    """Return (save_checkpoint, load_checkpoint) from Fidan's real module.

    Raises MissingInterfaceError if Fidan's module isn't present -- there is
    no fixture fallback for checkpoint integration tests against real
    training code; use tests/integration/test_checkpoint_integration.py's
    own minimal reference loop instead when Fidan doesn't exist yet.
    """
    # TODO(integration): point this at Fidan's real checkpoint module, e.g.:
    #   from training.checkpoint import save_checkpoint, load_checkpoint
    #   return save_checkpoint, load_checkpoint
    try:
        mod = importlib.import_module("src.training.checkpoint")
        return mod.save_checkpoint, mod.load_checkpoint
    except ModuleNotFoundError as exc:
        raise MissingInterfaceError(
            "Fidan's training.checkpoint.{save_checkpoint,load_checkpoint} is "
            "not importable yet."
        ) from exc


def manifest_hash_fn():
    """Return Ibrahim's sha256_file(path) -> str.

    Confirmed 2026-08-31 by reading Ibrahim's actual src/data/hashing.py:
    `sha256_file(path: Path, chunk_size: int = 8*1024*1024) -> str`,
    hex digest via hashlib.sha256(...).hexdigest(). The module is
    importable as `src.data.hashing` (Ibrahim's `src/` has an `__init__.py`,
    so it's a real package rooted at the repo root, not `data.hashing`
    as originally guessed).
    """
    try:
        mod = importlib.import_module("src.data.hashing")
        return mod.sha256_file
    except ModuleNotFoundError as exc:
        raise MissingInterfaceError(
            "Ibrahim's src.data.hashing.sha256_file is not importable. Run from "
            "the repository root so 'src' is importable as a top-level package."
        ) from exc


def load_training_data_contract(repo_root: str | Path = ".") -> dict:
    """Load Ibrahim's authoritative data/tokenizer handoff contract.

    Confirmed 2026-08-31 by reading the actual repo: Ibrahim publishes a single
    JSON file, `data/metadata/training_data_contract.json`, containing
    every artifact path + sha256 + byte count, `splits.seed`,
    `tokenizer.vocab_size`, and `tokenizer.artifact_hashes`. This
    supersedes an earlier (wrong) assumption of a `hash_manifest()`
    function and JSONL manifests -- Ibrahim's own README says "Yasin/Fidan should
    read data/metadata/training_data_contract.json", so this suite treats
    it as the primary interface too, rather than re-deriving hashes itself.
    """
    path = Path(repo_root) / "data" / "metadata" / "training_data_contract.json"
    if not path.is_file():
        raise MissingInterfaceError(
            f"Ibrahim's training data contract not found at {path}. Expected "
            "data/metadata/training_data_contract.json (see Ibrahim's README)."
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
    """Seed for the 50M-token training subset (training_subset.data_seed)."""
    try:
        return contract["training_subset"]["data_seed"]
    except KeyError as exc:
        raise MissingInterfaceError("training_data_contract.json missing training_subset.data_seed") from exc


def training_subset_hash_from_contract(contract: dict) -> str:
    """sha256 of data/manifests/train_50m.parquet (the actual training subset)."""
    try:
        return contract["training_subset"]["manifest_sha256"]
    except KeyError as exc:
        raise MissingInterfaceError("training_data_contract.json missing training_subset.manifest_sha256") from exc
