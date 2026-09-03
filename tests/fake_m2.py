"""SYNTHETIC_FIXTURE: a minimal test double for Yasin's `src.models` package.

Used ONLY to test that this suite's thin wrapper scripts (audit_configs.py,
audit_parameters.py) correctly plumb through whatever `src.models`
provides -- NOT to test Yasin's actual scientific logic (config-contract
correctness, parameter fairness, initialization parity), which is Yasin's
own responsibility and is covered by Yasin's own
tests/test_config_contract.py and tests/test_params_fairness.py.

This double intentionally implements the fairness/allowlist logic itself
(so tests can plant deliberate violations), but it is never used outside
this test file and never presented as a real fairness result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


FAKE_MODELS_SOURCE = '''
"""SYNTHETIC_FIXTURE test double for src.models (see tests/fake_m2.py)."""
import json
from pathlib import Path

PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")
ARM_ALLOWLIST = frozenset({"pe_type"})


class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def with_seed(self, seed):
        payload = dict(self.__dict__)
        payload["init_seed"] = seed
        return ModelConfig(**payload)

    def to_dict(self):
        return dict(self.__dict__)


def load_run_matrix(path):
    return json.loads(Path(path).read_text())


def fairness_report(base_config, pe_types=PE_TYPES):
    rows = []
    for pe in pe_types:
        cfg = base_config.to_dict()
        cfg["pe_type"] = pe
        # toy "parameter count": deterministic function of dims, +N for learned
        core = cfg["n_layer"] * cfg["d_model"] * cfg["d_ff"]
        positional = cfg["max_seq_len"] * cfg["d_model"] if pe == "learned" else 0
        rows.append({
            "pe_type": pe,
            "total": core + positional,
            "core": core,
            "positional": positional,
            "init_fingerprint": "fakehash",
        })
    # Test-only escape hatch: a planted config field lets tests exercise the
    # "violation detected" path without needing a real cross-arm mismatch,
    # since (like the real Yasin fairness_report) every arm here is derived
    # from the *same* base_config and so cannot organically disagree on
    # "core" except through this fixture's own injected fault.
    if base_config.to_dict().get("meta", {}).get("inject_violation"):
        rows[-1]["core"] += 1
    core_counts = {r["core"] for r in rows}
    violations = []
    if len(core_counts) != 1:
        violations.append(f"core parameter counts differ: {core_counts}")
    return {"rows": rows, "violations": violations, "passed": not violations}


def format_fairness_table(report):
    return "\\n".join(f"{r['pe_type']}: total={r['total']}" for r in report["rows"])
'''


def install_fake_models_package(repo_root: Path) -> None:
    """Swap in a synthetic `src.models` module, in memory only, for the
    duration of the calling test.

    This repo's real `src` package (src.data, src.tokenizer,
    src.reproducibility, ...) is left completely untouched -- only the
    `models` submodule is replaced, and only in sys.modules / as an
    attribute of the real `src` package object, never on disk. This lets
    these wrapper-plumbing tests run against a fake Yasin without disturbing
    every other test's real src.data/src.models/src.reproducibility imports.
    tests/unit/conftest.py restores whatever was in `src.models` before
    (the real module, in this repo) after every test.
    """
    import types

    import src  # the repo's real top-level package

    fake_module = types.ModuleType("src.models")
    fake_module.__file__ = str(repo_root / "src" / "models" / "__init__.py")
    exec(compile(FAKE_MODELS_SOURCE, fake_module.__file__, "exec"), fake_module.__dict__)  # noqa: S102

    sys.modules["src.models"] = fake_module
    src.models = fake_module


def write_fake_configs(repo_root: Path, payloads: dict[str, dict], run_seeds=(17, 42, 1234)) -> None:
    pe_dir = repo_root / "configs" / "pe"
    pe_dir.mkdir(parents=True, exist_ok=True)
    for pe, payload in payloads.items():
        (pe_dir / f"{pe}.json").write_text(json.dumps(payload))

    base = dict(next(iter(payloads.values())))
    base["pe_type"] = "nope"
    (repo_root / "configs" / "model_base.json").write_text(json.dumps(base))

    (repo_root / "configs" / "run_matrix.json").write_text(
        json.dumps(
            {
                "data_seed": 2026,
                "pe_types": list(payloads),
                "run_seeds": list(run_seeds),
                "runs": [{"pe_type": pe, "init_seed": s} for s in run_seeds for pe in payloads],
            }
        )
    )
