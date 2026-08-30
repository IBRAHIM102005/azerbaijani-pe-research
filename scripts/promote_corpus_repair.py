"""Promote a validated near-duplicate repair while preserving prerepair artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, canonical_json_hash, sha256_file


EXPECTED_CANDIDATES = 6_444_499
STALE_METADATA = (
    "preparation_summary.json",
    "near_duplicate_report.json",
    "manifest_hashes.json",
    "leakage_audit.json",
    "frozen_corpus_validation.json",
    "tokenizer_audit.json",
    "document_token_counts.parquet",
    "token_counts_by_source_split.json",
    "training_subset_summary.json",
    "training_data_contract.json",
    "corpus_release_status.json",
)


def candidate_count(path: Path) -> int:
    connection = sqlite3.connect("file:" + path.resolve().as_posix() + "?mode=ro", uri=True)
    try:
        return connection.execute("SELECT COUNT(*) FROM near_candidates").fetchone()[0]
    finally:
        connection.close()


def remove_inactive_sidecars(database: Path) -> None:
    wal = database.with_name(database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f"Refusing to move database with a non-empty WAL: {wal}")
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists():
            sidecar.unlink()


def parquet_rows(directory: Path) -> int:
    return sum(
        pq.ParquetFile(directory / f"{split}.parquet").metadata.num_rows
        for split in ("train", "validation", "test")
    )


def move_once(current: Path, backup: Path, staged: Path, expected_rows: int) -> None:
    if current.exists() and parquet_rows(current) == expected_rows:
        return
    if not staged.exists() or parquet_rows(staged) != expected_rows:
        raise RuntimeError(f"Validated staged directory is unavailable: {staged}")
    if current.exists():
        if backup.exists():
            raise RuntimeError(f"Both current and backup paths exist before promotion: {current}")
        os.replace(current, backup)
    os.replace(staged, current)


def main() -> None:
    config = load_config(repo_root=ROOT)
    metadata = config.path("metadata")
    validation = json.loads((metadata / "near_repair_validation.json").read_text(encoding="utf-8"))
    large = json.loads(
        (metadata / "near_repair_large_bucket_leakage.json").read_text(encoding="utf-8")
    )
    recall = large["audit"]["measured_candidate_recall"]
    cross_split = large["audit"]["cross_split_retained_representative_pairs"]
    if not validation.get("hard_gate_passed") or recall != 1.0 or cross_split != 0:
        raise RuntimeError("Independent repair validation has not passed")
    expected_rows = validation["retained_documents"]

    interim = config.path("interim")
    current_database = interim / "corpus_index.sqlite"
    staged_database = interim / "corpus_index_repair.sqlite"
    backup_database = interim / "corpus_index_prerepair.sqlite"
    if candidate_count(current_database) != EXPECTED_CANDIDATES:
        if backup_database.exists():
            raise RuntimeError("Prerepair database backup already exists before database promotion")
        if candidate_count(staged_database) != EXPECTED_CANDIDATES:
            raise RuntimeError("Staged database candidate count changed after validation")
        remove_inactive_sidecars(current_database)
        remove_inactive_sidecars(staged_database)
        os.replace(current_database, backup_database)
        os.replace(staged_database, current_database)

    processed = config.path("processed")
    move_once(
        processed,
        processed.with_name("corpus_prerepair"),
        processed.with_name("corpus_repair"),
        expected_rows,
    )

    manifests = config.path("manifests")
    staged_manifests = manifests.with_name("manifests_repair")
    backup_manifests = manifests.with_name("manifests_prerepair")
    current_manifest_rows = parquet_rows(manifests)
    if current_manifest_rows != expected_rows:
        if parquet_rows(staged_manifests) != expected_rows:
            raise RuntimeError("Staged manifest row count changed after validation")
        backup_manifests.mkdir(parents=True, exist_ok=True)
        for name in ("train.parquet", "validation.parquet", "test.parquet", "train_50m.parquet"):
            source = manifests / name
            if source.exists():
                destination = backup_manifests / name
                if destination.exists():
                    raise RuntimeError(f"Prerepair manifest already exists: {destination}")
                os.replace(source, destination)
        for split in ("train", "validation", "test"):
            os.replace(staged_manifests / f"{split}.parquet", manifests / f"{split}.parquet")
        staged_manifests.rmdir()

    prerepair_metadata = metadata / "prerepair"
    prerepair_metadata.mkdir(parents=True, exist_ok=True)
    old_preparation = None
    for name in STALE_METADATA:
        source = metadata / name
        destination = prerepair_metadata / name
        if source.exists() and not destination.exists():
            if name == "preparation_summary.json":
                old_preparation = json.loads(source.read_text(encoding="utf-8"))
            os.replace(source, destination)
    if old_preparation is None:
        old_preparation = json.loads(
            (prerepair_metadata / "preparation_summary.json").read_text(encoding="utf-8")
        )

    stage = json.loads((metadata / "near_repair_stage_summary.json").read_text(encoding="utf-8"))
    final_hashes = {
        f"data/manifests/{Path(name).name}": digest
        for name, digest in stage["manifest_hashes"].items()
    }
    near = {
        **stage["near_duplicates"],
        "candidate_generation": stage["candidate_generation"],
        "fingerprinted_documents": 6_209_184,
        "connected_component_semantics": (
            "Every accepted edge has direct character-5-gram Jaccard >= 0.95. "
            "Clusters are connected components and are not necessarily pairwise >= 0.95."
        ),
        "supersedes_prerepair_report": "data/metadata/prerepair/near_duplicate_report.json",
    }
    preparation = old_preparation
    preparation["config_sha256"] = canonical_json_hash(config.values)
    preparation["near_duplicates"] = near
    preparation["split_summary"] = stage["split_summary"]
    preparation["manifest_hashes"] = final_hashes
    accounting = preparation["corpus_accounting"]
    accounting.update(
        {
            "near_candidate_pairs": near["candidate_pairs_checked"],
            "near_accepted_edges": near["accepted_edges"],
            "near_clusters": near["clusters"],
            "near_clustered_documents": near["clustered_documents"],
            "near_duplicate_removals": near["removed_documents"],
            "final_retained_documents": expected_rows,
        }
    )
    atomic_write_json(metadata / "near_duplicate_report.json", near)
    atomic_write_json(metadata / "manifest_hashes.json", final_hashes)
    atomic_write_json(metadata / "preparation_summary.json", preparation)
    leakage = {
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "layers": {
            "internal_manifest_and_graph": validation,
            "independent_all_large_bucket_audit": large,
        },
        "former_237_pairs_unresolved": validation["prerepair_237_pair_regression"][
            "unresolved_cross_split"
        ],
        "manifest_hashes": final_hashes,
    }
    atomic_write_json(metadata / "leakage_audit.json", leakage)
    promotion = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "database": "data/interim/corpus/corpus_index.sqlite",
        "prerepair_database": "data/interim/corpus/corpus_index_prerepair.sqlite",
        "retained_documents": expected_rows,
        "candidate_pairs": EXPECTED_CANDIDATES,
        "manifest_hashes": final_hashes,
        "leakage_audit_sha256": sha256_file(metadata / "leakage_audit.json"),
    }
    atomic_write_json(metadata / "near_repair_promotion.json", promotion)
    print(json.dumps(promotion, indent=2))


if __name__ == "__main__":
    main()
