"""Read the frozen M1 identity from ``training_data_contract.json``.

Every tokenizer path, artifact hash and token-budget number used by M2 is read
from the contract at run time.  Nothing here re-states a hash as a literal: a
duplicated constant is exactly how two sources of truth drift apart, and the
contract is the one M1 froze.

M2 only ever *reads* these files.  Nothing in this module writes, regenerates
or modifies an M1 artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "data" / "metadata" / "training_data_contract.json"

__all__ = [
    "DataContract",
    "load_contract",
    "sha256_file",
]


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DataContract:
    """Typed view over the parts of the contract that M2 depends on."""

    raw: Dict[str, Any]
    root: Path = REPO_ROOT

    # --- tokenizer ---------------------------------------------------------
    @property
    def tokenizer_path(self) -> str:
        return self.raw["artifacts"]["tokenizer"]["tokenizer.model"]["path"]

    @property
    def tokenizer_sha256(self) -> str:
        return self.raw["tokenizer"]["artifact_hashes"]["tokenizer.model"]

    @property
    def vocab_size(self) -> int:
        return int(self.raw["tokenizer"]["vocab_size"])

    @property
    def eod_id(self) -> int:
        return int(self.raw["tokenizer"]["special_tokens"]["eod"]["id"])

    # --- splits / manifests ------------------------------------------------
    @property
    def manifest_hashes(self) -> Dict[str, str]:
        return {
            split: entry["sha256"]
            for split, entry in self.raw["splits"]["manifests"].items()
        }

    @property
    def manifest_paths(self) -> Dict[str, str]:
        return {
            split: entry["path"]
            for split, entry in self.raw["splits"]["manifests"].items()
        }

    # --- the 50M training subset ------------------------------------------
    @property
    def training_subset_sha256(self) -> str:
        return self.raw["training_subset"]["manifest_sha256"]

    @property
    def training_subset_path(self) -> str:
        return self.raw["training_subset"]["manifest_path"]

    @property
    def target_tokens(self) -> int:
        return int(self.raw["training_subset"]["target_tokens"])

    @property
    def selected_tokens(self) -> int:
        return int(self.raw["training_subset"]["selected_unique_tokens"])

    @property
    def data_seed(self) -> int:
        return int(self.raw["training_subset"]["data_seed"])

    @property
    def model_seed_affects_order(self) -> bool:
        return bool(self.raw["training_subset"]["model_seed_affects_order"])

    # --- helpers -----------------------------------------------------------
    def artifact_exists(self, relative_path: str) -> bool:
        return (self.root / relative_path).is_file()

    def verify(self, relative_path: str, expected_sha256: str) -> Optional[bool]:
        """``True``/``False`` if the file is present, ``None`` if it is absent.

        The large parquet artifacts are not committed to git, so callers treat
        ``None`` as "cannot check here" rather than as a failure.
        """
        target = self.root / relative_path
        if not target.is_file():
            return None
        return sha256_file(target) == expected_sha256


def load_contract(path: Path | str = CONTRACT_PATH) -> DataContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DataContract(raw=payload)
