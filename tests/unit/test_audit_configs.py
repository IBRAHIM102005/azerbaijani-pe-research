import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import all_arm_payloads, mutate  
from fake_m2 import install_fake_models_package, write_fake_configs  
import audit_configs  


def _setup(tmp_path: Path, payloads: dict) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    install_fake_models_package(repo_root)
    write_fake_configs(repo_root, payloads)
    return repo_root


def test_all_allowed_differences_pass(tmp_path):
    repo_root = _setup(tmp_path, all_arm_payloads())
    report = audit_configs.run_audit(repo_root)
    assert report["forbidden_total"] == 0, report


def test_only_pe_type_is_allowlisted(tmp_path):
    repo_root = _setup(tmp_path, all_arm_payloads())
    report = audit_configs.run_audit(repo_root)
    assert report["allowlist"] == ["pe_type"]
    assert "pe_type" in report["allowed_differences"]


def test_hidden_dim_drift_fails(tmp_path):
    payloads = all_arm_payloads()
    payloads["alibi"] = mutate(payloads["alibi"], "d_model", 999)
    repo_root = _setup(tmp_path, payloads)
    report = audit_configs.run_audit(repo_root)
    assert report["forbidden_total"] > 0
    assert "d_model" in report["forbidden_differences"]


def test_learning_relevant_field_drift_fails(tmp_path):
    payloads = all_arm_payloads()
    payloads["nope"] = mutate(payloads["nope"], "rotary_pct", 0.9)
    repo_root = _setup(tmp_path, payloads)
    report = audit_configs.run_audit(repo_root)
    assert report["forbidden_total"] > 0
    assert "rotary_pct" in report["forbidden_differences"]


def test_missing_pe_variant_errors(tmp_path):
    payloads = all_arm_payloads()
    del payloads["nope"]
    repo_root = _setup(tmp_path, payloads)
    with pytest.raises(SystemExit):
        audit_configs.run_audit(repo_root)


def test_missing_repo_root_errors(tmp_path):
    with pytest.raises(SystemExit):
        audit_configs.run_audit(tmp_path / "does_not_exist")


def test_m2_not_importable_errors_loudly(tmp_path):
    repo_root = tmp_path / "no_m2_repo"
    (repo_root / "configs" / "pe").mkdir(parents=True)
    import sys as _sys

    for name in list(_sys.modules):
        if name == "src" or name.startswith("src."):
            del _sys.modules[name]
    _sys.path[:] = [str(repo_root)]
    with pytest.raises(SystemExit):
        audit_configs.run_audit(repo_root)
