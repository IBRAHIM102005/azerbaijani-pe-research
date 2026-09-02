import json
import sys
from pathlib import Path

import pytest


from src.reproducibility.adapters import (  
    MissingInterfaceError,
    data_seed_from_contract,
    load_training_data_contract,
    manifest_hashes_from_contract,
    tokenizer_hashes_from_contract,
    tokenizer_vocab_size_from_contract,
)
from src.reproducibility.config_utils import combined_hash  
MINIMAL_CONTRACT = {
    "m1_status": "complete",
    "splits": {"seed": 2026},
    "tokenizer": {
        "vocab_size": 16000,
        "artifact_hashes": {
            "tokenizer.model": "a" * 64,
            "tokenizer.vocab": "b" * 64,
            "tokenizer_config.json": "c" * 64,
            "special_tokens.json": "d" * 64,
        },
    },
    "artifacts": {
        "manifests": {
            "train": {"path": "data/manifests/train.parquet", "sha256": "e" * 64, "bytes": 123},
            "validation": {"path": "data/manifests/validation.parquet", "sha256": "f" * 64, "bytes": 45},
            "test": {"path": "data/manifests/test.parquet", "sha256": "0" * 64, "bytes": 67},
        }
    },
}


def _write_contract(tmp_path: Path, contract: dict) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "metadata").mkdir(parents=True)
    (repo_root / "data" / "metadata" / "training_data_contract.json").write_text(json.dumps(contract))
    return repo_root


def test_load_training_data_contract(tmp_path):
    repo_root = _write_contract(tmp_path, MINIMAL_CONTRACT)
    contract = load_training_data_contract(repo_root)
    assert contract["m1_status"] == "complete"


def test_load_training_data_contract_missing_raises(tmp_path):
    with pytest.raises(MissingInterfaceError):
        load_training_data_contract(tmp_path / "nonexistent_repo")


def test_manifest_hashes_from_contract(tmp_path):
    hashes = manifest_hashes_from_contract(MINIMAL_CONTRACT)
    assert hashes == {"train": "e" * 64, "validation": "f" * 64, "test": "0" * 64}


def test_manifest_hashes_from_contract_missing_split_raises():
    broken = json.loads(json.dumps(MINIMAL_CONTRACT))
    del broken["artifacts"]["manifests"]["test"]
    with pytest.raises(MissingInterfaceError):
        manifest_hashes_from_contract(broken)


def test_tokenizer_hashes_from_contract():
    hashes = tokenizer_hashes_from_contract(MINIMAL_CONTRACT)
    assert set(hashes) == {"tokenizer.model", "tokenizer.vocab", "tokenizer_config.json", "special_tokens.json"}


def test_tokenizer_vocab_size_and_data_seed_from_contract():
    assert tokenizer_vocab_size_from_contract(MINIMAL_CONTRACT) == 16000
    assert data_seed_from_contract(MINIMAL_CONTRACT) == 2026


def test_combined_hash_is_stable_and_order_independent():
    h1 = combined_hash({"a": "1", "b": "2"})
    h2 = combined_hash({"b": "2", "a": "1"})
    assert h1 == h2
    assert len(h1) == 64


def test_combined_hash_changes_with_content():
    h1 = combined_hash({"a": "1"})
    h2 = combined_hash({"a": "2"})
    assert h1 != h2
