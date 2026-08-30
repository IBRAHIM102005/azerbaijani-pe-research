"""Consolidate final repaired hashes and layered validation evidence."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, sha256_file
from src.data.paths import repository_relative


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = load_config(repo_root=ROOT)
    metadata = config.path("metadata")
    leakage_path = metadata / "leakage_audit.json"
    leakage = load(leakage_path)
    downstream = load(metadata / "downstream_repair_validation.json")
    raw = load(metadata / "raw_immutability.json")
    if leakage["status"] != "pass" or downstream["status"] != "pass" or raw["status"] != "pass":
        raise RuntimeError("A repaired data pipeline safety layer is not passing")
    leakage["layers"]["downstream_train_only_and_sequence_replay"] = downstream
    leakage["raw_immutability_status"] = raw["status"]
    leakage["final_status"] = "pass"
    atomic_write_json(leakage_path, leakage)

    processed_hashes = {}
    for split in ("train", "validation", "test"):
        path = config.path("processed") / f"{split}.parquet"
        processed_hashes[repository_relative(path, ROOT)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_write_json(metadata / "processed_hashes.json", processed_hashes)

    artifacts = {
        "near_repair_candidate_recall": metadata / "near_repair_candidate_recall.json",
        "near_repair_all_large_bucket_audit": metadata / "near_repair_large_bucket_leakage.json",
        "near_repair_validation": metadata / "near_repair_validation.json",
        "downstream_repair_validation": metadata / "downstream_repair_validation.json",
        "leakage_audit": leakage_path,
        "raw_immutability": metadata / "raw_immutability.json",
        "tokenizer_audit": metadata / "tokenizer_audit.json",
        "token_counts": metadata / "document_token_counts.parquet",
        "training_subset": config.path("manifests") / "train_50m.parquet",
        "pytest_junit": metadata / "pytest_data_pipeline.xml",
        "m1_validation": metadata / "frozen_corpus_validation.json",
    }
    final = {
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_hashes": {
            name: {
                "path": repository_relative(path, ROOT),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
        "processed_hashes": processed_hashes,
        "portability_policy": (
            "Repository-internal operational paths are POSIX-style paths relative to the repository root."
        ),
    }
    atomic_write_json(metadata / "repair_final_metadata.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
