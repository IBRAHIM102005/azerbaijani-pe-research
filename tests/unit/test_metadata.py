import json
import sys
from pathlib import Path

import pytest
import yaml

from src.reproducibility.config_utils import config_hash
from src.reproducibility.metadata import collect_metadata, device_info, timestamps, write_metadata, UNAVAILABLE  # noqa: E402


def test_collect_metadata_works_cpu_only():
    meta = collect_metadata(
        run_id="test-run",
        pe_method="nope",
        model_seed=17,
        data_seed=2026,
        resolved_config_hash="deadbeef" * 8,
    )
    assert meta["run_id"] == "test-run"
    assert meta["pe_method"] == "nope"
    assert meta["model_seed"] == 17
    assert meta["data_seed"] == 2026
    # unsupplied fields must be explicit, not missing keys
    for field in [
        "tokenizer_hash",
        "train_manifest_hash",
        "validation_manifest_hash",
        "test_manifest_hash",
        "training_subset_manifest_hash",
        "training_cache_hash",
        "dataset_source_revision",
        "precision",
    ]:
        assert meta[field] == UNAVAILABLE


def test_python_version_always_available():
    # platform.python_version() can never fail; unlike numpy/torch this
    # should never be UNAVAILABLE.
    info = device_info()
    assert info["python_version"] != UNAVAILABLE
    assert info["python_version"][0].isdigit()


def test_metadata_is_valid_json_with_stable_keys(tmp_path):
    meta = collect_metadata(
        run_id="test-run-2",
        pe_method="rope",
        model_seed=42,
        data_seed=2026,
        resolved_config_hash="cafebabe" * 8,
        tokens_seen=123456,
        exit_code=0,
    )
    out_path = tmp_path / "meta.json"
    write_metadata(meta, out_path)
    reloaded = json.loads(out_path.read_text())
    assert reloaded == meta
    assert list(reloaded.keys()) == list(meta.keys())


def test_device_info_never_raises_without_cuda():
    info = device_info()
    assert "device_name" in info
    assert "cuda_version" in info
    assert "pytorch_version" in info
    assert "python_version" in info
    assert "numpy_version" in info
    # must be a real string/value, never None or missing
    for v in info.values():
        assert v is not None


def test_timestamps_include_utc_and_baku():
    ts = timestamps()
    assert ts["timestamp_utc"].endswith("+00:00")
    assert "+04:00" in ts["timestamp_baku"]


def test_missing_git_repo_reports_unavailable_not_crash(tmp_path):
    meta = collect_metadata(
        run_id="no-git",
        pe_method="alibi",
        model_seed=1234,
        data_seed=2026,
        resolved_config_hash="abc123ab" * 8,
        repo_dir=str(tmp_path),  # not a git repo
    )
    assert meta["git_commit"] == UNAVAILABLE
    assert meta["dirty_repository"] == UNAVAILABLE


def test_unsafe_run_id_is_rejected():
    with pytest.raises(ValueError, match="run_id"):
        collect_metadata(
            run_id="../escape",
            pe_method="nope",
            model_seed=1,
            data_seed=2026,
            resolved_config_hash="deadbeef" * 8,
        )


def test_empty_run_id_is_rejected():
    with pytest.raises(ValueError, match="run_id"):
        collect_metadata(
            run_id="",
            pe_method="nope",
            model_seed=1,
            data_seed=2026,
            resolved_config_hash="deadbeef" * 8,
        )


def test_non_hash_shaped_config_hash_is_rejected():
    with pytest.raises(ValueError, match="resolved_config_hash"):
        collect_metadata(
            run_id="test-run",
            pe_method="nope",
            model_seed=1,
            data_seed=2026,
            resolved_config_hash="not-a-real-hash",
        )


def test_config_hash_mismatch_against_real_resolved_file_is_rejected(tmp_path):
    resolved_dir = tmp_path / "configs" / "frozen" / "resolved"
    resolved_dir.mkdir(parents=True)
    resolved_config = {"pe_type": "rope", "init_seed": 17}
    (resolved_dir / "some-run.yaml").write_text(yaml.safe_dump(resolved_config))

    with pytest.raises(ValueError, match="does not match"):
        collect_metadata(
            run_id="some-run",
            pe_method="rope",
            model_seed=17,
            data_seed=2026,
            resolved_config_hash="deadbeef" * 8,  # wrong on purpose
            repo_dir=str(tmp_path),
        )


def test_config_hash_matching_real_resolved_file_is_accepted(tmp_path):
    resolved_dir = tmp_path / "configs" / "frozen" / "resolved"
    resolved_dir.mkdir(parents=True)
    resolved_config = {"pe_type": "rope", "init_seed": 17}
    (resolved_dir / "some-run.yaml").write_text(yaml.safe_dump(resolved_config))
    real_hash = config_hash(resolved_config)

    meta = collect_metadata(
        run_id="some-run",
        pe_method="rope",
        model_seed=17,
        data_seed=2026,
        resolved_config_hash=real_hash,
        repo_dir=str(tmp_path),
    )
    assert meta["resolved_config_hash"] == real_hash


def test_training_subset_hash_and_vram_overrides_are_recorded():
    meta = collect_metadata(
        run_id="test-run",
        pe_method="nope",
        model_seed=17,
        data_seed=2026,
        resolved_config_hash="deadbeef" * 8,
        training_subset_manifest_hash="805f7b18" * 8,
        training_cache_hash="cafed00d" * 8,
        peak_allocated_vram_bytes=123456,
        peak_reserved_vram_bytes=234567,
    )
    assert meta["training_subset_manifest_hash"] == "805f7b18" * 8
    assert meta["training_cache_hash"] == "cafed00d" * 8
    assert meta["peak_allocated_vram_bytes"] == 123456
    assert meta["peak_reserved_vram_bytes"] == 234567
