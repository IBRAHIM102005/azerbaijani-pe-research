import pytest

from src.data.config import DataPipelineConfig
from src.data.paths import repository_relative, resolve_repository_path


def test_relative_manifest_path_resolves_under_a_new_repo_root(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "moved"
    first_path = first_root / "data" / "processed" / "corpus" / "train.parquet"
    reference = repository_relative(first_path, first_root)
    assert reference == "data/processed/corpus/train.parquet"
    assert resolve_repository_path(reference, second_root) == (
        second_root / "data" / "processed" / "corpus" / "train.parquet"
    ).resolve()


def test_path_resolver_rejects_escape_and_absolute_paths(tmp_path):
    with pytest.raises(ValueError):
        resolve_repository_path("../outside.parquet", tmp_path)
    with pytest.raises(ValueError):
        resolve_repository_path(str((tmp_path / "absolute.parquet").resolve()), tmp_path)


def test_dollma_root_prefers_explicit_environment_override(tmp_path, monkeypatch):
    configured = tmp_path / "configured"
    override = tmp_path / "external-dollma"
    override.mkdir()
    config = DataPipelineConfig(
        repo_root=tmp_path,
        config_path=tmp_path / "data_pipeline.yaml",
        values={"paths": {"original_dollma": configured.name}},
    )
    monkeypatch.setenv("AZ_PE_DOLLMA_ROOT", str(override))
    assert config.path("original_dollma") == override.resolve()


def test_missing_dollma_root_has_actionable_error(tmp_path, monkeypatch):
    config = DataPipelineConfig(
        repo_root=tmp_path,
        config_path=tmp_path / "data_pipeline.yaml",
        values={"paths": {"original_dollma": "missing-dollma"}},
    )
    monkeypatch.delenv("AZ_PE_DOLLMA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="Set AZ_PE_DOLLMA_ROOT"):
        config.path("original_dollma")
