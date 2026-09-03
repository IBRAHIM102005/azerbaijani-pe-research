import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import all_arm_payloads  # noqa: E402
from fake_m2 import install_fake_models_package, write_fake_configs  # noqa: E402
import verify_release  # noqa: E402


def _make_fake_repo(tmp_path: Path, complete: bool) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    install_fake_models_package(root)
    write_fake_configs(root, all_arm_payloads())
    (root / "README.md").write_text("# fake repo\n")
    (root / "Makefile").write_text("test:\n\techo ok\n")
    (root / "data" / "metadata").mkdir(parents=True)
    (root / "data" / "README.md").write_text("data access docs\n")
    (root / "data" / "metadata" / "source_registry.yaml").write_text("dataset: DOLLMA\n")
    (root / "tokenizer").mkdir()
    tokenizer_hashes = {
        "tokenizer.model": "a" * 64,
        "tokenizer.vocab": "b" * 64,
        "tokenizer_config.json": "c" * 64,
        "special_tokens.json": "d" * 64,
    }
    (root / "tokenizer" / "tokenizer_hashes.json").write_text(json.dumps(tokenizer_hashes))
    contract = {
        "m1_status": "complete",
        "splits": {"seed": 2026},
        "training_subset": {
            "data_seed": 2026,
            "manifest_path": "data/manifests/train_50m.parquet",
            "manifest_sha256": "1" * 64,
        },
        "tokenizer": {"vocab_size": 16000, "artifact_hashes": tokenizer_hashes},
        "artifacts": {
            "manifests": {
                "train": {"path": "data/manifests/train.parquet", "sha256": "e" * 64},
                "validation": {"path": "data/manifests/validation.parquet", "sha256": "f" * 64},
                "test": {"path": "data/manifests/test.parquet", "sha256": "0" * 64},
            }
        },
    }
    (root / "data" / "metadata" / "training_data_contract.json").write_text(json.dumps(contract))
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "experiments" / "manifests").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs" / "ai_use.md").write_text("no substantial AI use beyond what is disclosed\n")

    if not complete:
        (root / "docs" / "ai_use.md").unlink()  # remove a required artifact

    return root


def test_complete_repo_passes_artifact_and_config_and_param_checks(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    assert result["artifact_verification"]["pass"] is True
    assert result["config_audit"]["pass"] is True
    assert result["parameter_audit"]["pass"] is True
    assert result["overall_pass"] is True


def test_missing_required_artifact_fails_clearly(tmp_path):
    root = _make_fake_repo(tmp_path, complete=False)
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    assert result["artifact_verification"]["pass"] is False
    assert result["overall_pass"] is False
    missing_paths = [
        i["path"]
        for i in result["artifact_verification"]["items"]
        if i["required"] and not i["present"]
    ]
    assert "docs/ai_use.md" in missing_paths


def test_missing_resolved_configs_dir_fails_clearly(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    import shutil

    shutil.rmtree(root / "configs")
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    assert result["config_audit"]["pass"] is False
    assert result["parameter_audit"]["pass"] is False
    assert result["overall_pass"] is False


def test_report_contains_required_top_level_fields(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    for field in [
        "overall_pass",
        "timestamp_utc",
        "git_commit",
        "artifact_verification",
        "checkpoint_interface",
        "config_audit",
        "parameter_audit",
        "test_result",
        "training_data_contract",
    ]:
        assert field in result


def test_training_data_contract_check_passes_when_complete(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    tdc = result["training_data_contract"]
    assert tdc["pass"] is True
    assert tdc["m1_status"] == "complete"
    assert tdc["vocab_size"] == 16000
    assert tdc["data_seed"] == 2026
    assert tdc["training_subset_hash"] == "1" * 64
    # Files aren't present on disk in this fixture repo; must be skipped, not failed.
    assert set(tdc["skipped_not_present"]) >= {
        "data/manifests/train.parquet",
        "tokenizer/tokenizer.model",
        "data/manifests/train_50m.parquet",
    }
    assert tdc["mismatched"] == []


def test_training_data_contract_checks_50m_subset_hash_not_just_full_split(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    # Must be caught independently of the full train.parquet split.
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifests" / "train_50m.parquet").write_bytes(b"wrong subset bytes")
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    tdc = result["training_data_contract"]
    assert tdc["pass"] is False
    assert any(m["path"] == "data/manifests/train_50m.parquet" for m in tdc["mismatched"])


def test_training_data_contract_detects_on_disk_hash_mismatch(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    # Hash mismatch must be caught, not skipped.
    (root / "tokenizer" / "tokenizer.model").write_bytes(b"not the real tokenizer bytes")
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    tdc = result["training_data_contract"]
    assert tdc["pass"] is False
    assert any(m["path"] == "tokenizer/tokenizer.model" for m in tdc["mismatched"])
    assert result["overall_pass"] is False


def test_training_data_contract_verifies_matching_on_disk_hash(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    content = b"the real tokenizer bytes"
    import hashlib

    real_hash = hashlib.sha256(content).hexdigest()
    contract_path = root / "data" / "metadata" / "training_data_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["tokenizer"]["artifact_hashes"]["tokenizer.model"] = real_hash
    contract_path.write_text(json.dumps(contract))
    (root / "tokenizer" / "tokenizer_hashes.json").write_text(
        json.dumps(contract["tokenizer"]["artifact_hashes"])
    )
    (root / "tokenizer" / "tokenizer.model").write_bytes(content)

    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    tdc = result["training_data_contract"]
    assert "tokenizer/tokenizer.model" in tdc["verified_on_disk"]
    assert tdc["mismatched"] == []


def test_checkpoint_interface_reports_synthetic_fixture_honestly(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    ci = result["checkpoint_interface"]
    # Fidan doesn't exist here; must say so, but not block the release.
    assert ci["using_real_m3_checkpoint"] is False
    assert "SYNTHETIC_FIXTURE" in ci["note"]


def test_training_data_contract_missing_file_fails_clearly(tmp_path):
    root = _make_fake_repo(tmp_path, complete=True)
    (root / "data" / "metadata" / "training_data_contract.json").unlink()
    result = verify_release.run_verification(root, seed=None, skip_tests=True)
    assert result["training_data_contract"]["pass"] is False
    assert "error" in result["training_data_contract"]
    assert result["overall_pass"] is False
