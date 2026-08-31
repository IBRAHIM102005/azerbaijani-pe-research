"""Resolve a production run configuration, and give it a traceable identity.

Two seeds exist in this project and they are deliberately independent:

``data_seed``
    Fixed at 2026 by M1.  It determines the frozen 50M-token subset and its
    order.  ``training_data_contract.json`` records
    ``model_seed_affects_order = false``, so no model seed can perturb it.

``init_seed``
    The model initialisation seed, one of the preregistered
    ``RUN_SEEDS = (17, 42, 1234)``.  It comes from ``configs/run_matrix.json``
    and from nowhere else.

The shipped arm configs carry ``init_seed = -1`` (the template sentinel) and
``PELanguageModel`` refuses to build from them, so a template value cannot
become the seed of a real run by accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .config import PE_TYPES, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
RUN_MATRIX_PATH = CONFIG_DIR / "run_matrix.json"

#: Traceable run id, per the research master plan.
RUN_ID_TEMPLATE = "p31az-{pe}-s{seed}-t50m-c{ctx}-v{vocab}-{conf8}"

__all__ = [
    "ResolvedRun",
    "config_sha256",
    "load_run_matrix",
    "resolve_run_config",
    "iter_runs",
]


def config_sha256(config: ModelConfig) -> str:
    """Stable hash of the fully resolved config (sorted, compact JSON)."""
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedRun:
    """One runnable (arm, seed) cell of the experiment."""

    pe_type: str
    init_seed: int
    config: ModelConfig
    config_sha256: str
    run_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pe_type": self.pe_type,
            "init_seed": self.init_seed,
            "data_seed": self.config.data_seed,
            "config_sha256": self.config_sha256,
            "config": self.config.to_dict(),
        }


def load_run_matrix(path: Path | str = RUN_MATRIX_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_run_config(
    pe_type: str, init_seed: int, config_dir: Path | str = CONFIG_DIR
) -> ResolvedRun:
    """Load the arm template, stamp the seed, and derive the run identity."""
    if pe_type not in PE_TYPES:
        raise ValueError(f"unknown pe_type {pe_type!r}")

    matrix = load_run_matrix(Path(config_dir) / "run_matrix.json")
    if init_seed not in matrix["run_seeds"]:
        raise ValueError(
            f"{init_seed} is not a preregistered run seed {matrix['run_seeds']}; "
            "adding one requires an explicit protocol amendment"
        )

    template = ModelConfig.from_json(Path(config_dir) / "pe" / f"{pe_type}.json")
    if not template.is_template:
        raise ValueError(
            f"configs/pe/{pe_type}.json already carries init_seed="
            f"{template.init_seed}; shipped configs must stay templates"
        )

    config = template.with_seed(init_seed)
    digest = config_sha256(config)
    run_id = RUN_ID_TEMPLATE.format(
        pe=pe_type,
        seed=init_seed,
        ctx=config.max_seq_len,
        vocab=f"{config.vocab_size // 1000}k",
        conf8=digest[:8],
    )
    return ResolvedRun(pe_type, init_seed, config, digest, run_id)


def iter_runs(config_dir: Path | str = CONFIG_DIR) -> Iterator[ResolvedRun]:
    """Every (arm, seed) cell of the preregistered matrix, in matrix order."""
    matrix = load_run_matrix(Path(config_dir) / "run_matrix.json")
    for entry in matrix["runs"]:
        yield resolve_run_config(entry["pe_type"], entry["init_seed"], config_dir)


def run_ids(config_dir: Path | str = CONFIG_DIR) -> List[str]:
    return [run.run_id for run in iter_runs(config_dir)]
