import json
import sys
from pathlib import Path


from src.reproducibility.metadata import collect_metadata, device_info, timestamps, write_metadata, UNAVAILABLE  # noqa: E402


def test_collect_metadata_works_cpu_only():
    meta = collect_metadata(
        run_id="test-run",
        pe_method="nope",
        model_seed=17,
        data_seed=2026,
        resolved_config_hash="deadbeef",
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
        "dataset_source_revision",
        "precision",
    ]:
        assert meta[field] == UNAVAILABLE


def test_metadata_is_valid_json_with_stable_keys(tmp_path):
    meta = collect_metadata(
        run_id="test-run-2",
        pe_method="rope",
        model_seed=42,
        data_seed=2026,
        resolved_config_hash="cafebabe",
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
        resolved_config_hash="abc123",
        repo_dir=str(tmp_path),  # not a git repo
    )
    assert meta["git_commit"] == UNAVAILABLE
    assert meta["dirty_repository"] == UNAVAILABLE
