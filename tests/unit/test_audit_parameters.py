import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import all_arm_payloads, mutate  # noqa: E402
from fake_m2 import install_fake_models_package, write_fake_configs  # noqa: E402
import audit_parameters  # noqa: E402


def _setup(tmp_path: Path, payloads: dict, run_seeds=(17, 42, 1234, 2027, 5003)) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    install_fake_models_package(repo_root)
    write_fake_configs(repo_root, payloads, run_seeds=run_seeds)
    return repo_root


def test_fair_configs_pass(tmp_path):
    repo_root = _setup(tmp_path, all_arm_payloads())
    report = audit_parameters.run_audit(repo_root)
    assert report["passed"], report["violations"]
    assert report["resolved_seed"] == 17  # first preregistered seed by default


def test_seed_override_is_used(tmp_path):
    repo_root = _setup(tmp_path, all_arm_payloads())
    report = audit_parameters.run_audit(repo_root, seed=42)
    assert report["resolved_seed"] == 42


def test_unregistered_seed_rejected(tmp_path):
    repo_root = _setup(tmp_path, all_arm_payloads())
    with pytest.raises(SystemExit):
        audit_parameters.run_audit(repo_root, seed=99999)


def test_core_mismatch_between_arms_detected(tmp_path):
    payloads = all_arm_payloads(meta={"inject_violation": True})
    repo_root = _setup(tmp_path, payloads)
    report = audit_parameters.run_audit(repo_root)
    assert not report["passed"]
    assert any("differ" in v for v in report["violations"])


def test_missing_config_file_errors_clearly(tmp_path):
    repo_root = _setup(tmp_path, all_arm_payloads())
    with pytest.raises(SystemExit):
        audit_parameters.run_audit(repo_root, base_config_path=repo_root / "nope.json")


def test_m2_not_importable_errors_loudly(tmp_path):
    # This repo has a real src.models (Yasin is done), so simply deleting
    # cached modules isn't enough -- sys.path must also be restricted to
    # the empty fake repo, or a fresh import would just find the real one.
    repo_root = tmp_path / "no_m2_repo"
    repo_root.mkdir()
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path[:] = [str(repo_root)]
    with pytest.raises(SystemExit):
        audit_parameters.run_audit(repo_root)
