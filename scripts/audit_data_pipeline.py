"""Verify the frozen data pipeline corpus before tokenizer training and handoff."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, canonical_json_hash, sha256_file


LOGGER = logging.getLogger("data_pipeline.audit")
SPLITS = ("train", "validation", "test")
REQUIRED_MANIFEST_COLUMNS = {
    "document_id",
    "source",
    "source_group",
    "duplicate_cluster_id",
    "canonical_text_hash",
    "processed_file",
    "processed_row",
    "raw_record_id",
    "raw_shard",
    "raw_row_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen data pipeline accounting, manifests, and split leakage.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    return parser.parse_args()


def iter_column(path: Path, column: str, batch_size: int = 16_384) -> Iterable[str]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[column]):
        yield from batch.column(0).to_pylist()


def exact_report(metadata: Path, connection: sqlite3.Connection) -> dict:
    group_path = metadata / "exact_duplicate_groups.csv"
    member_path = metadata / "exact_duplicate_members.csv"
    groups = 0
    cross_source_groups = 0
    removal_sum = 0
    with group_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups += 1
            removal_sum += int(row["group_size"]) - 1
            sources = {item.strip() for item in row["sources"].split(",") if item.strip()}
            cross_source_groups += len(sources) > 1
    removed_documents = connection.execute("SELECT COUNT(*) FROM exact_removed").fetchone()[0]
    member_documents = 0
    with member_path.open("r", encoding="utf-8", newline="") as handle:
        member_documents = sum(1 for _ in csv.DictReader(handle))
    if removal_sum != removed_documents or member_documents != removed_documents:
        raise RuntimeError("Exact duplicate CSV accounting does not match the persisted database")
    return {
        "groups": groups,
        "removed_documents": removed_documents,
        "cross_source_groups": cross_source_groups,
        "group_report": str(group_path.resolve()),
        "member_report": str(member_path.resolve()),
    }


def validate_manifest_structure(config, summary: dict) -> tuple[dict, dict[str, Path]]:
    paths = {split: config.path("manifests") / f"{split}.parquet" for split in SPLITS}
    allowed_sources = set(config.included_sources)
    measured = {}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {path}")
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = REQUIRED_MANIFEST_COLUMNS - columns
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        source_counts = Counter()
        previous_id = None
        rows = 0
        for batch in parquet.iter_batches(
            batch_size=16_384,
            columns=["document_id", "source", "duplicate_cluster_id", "canonical_text_hash", "raw_record_id"],
        ):
            values = [column.to_pylist() for column in batch.columns]
            for document_id, source, cluster_id, canonical_hash, record_id in zip(*values):
                if not document_id or not cluster_id or not canonical_hash or not record_id:
                    raise RuntimeError(f"Empty required identifier in {split} manifest")
                if source not in allowed_sources:
                    raise RuntimeError(f"Unknown source {source!r} in {split} manifest")
                if previous_id is not None and document_id <= previous_id:
                    raise RuntimeError(f"Document IDs are not strictly sorted and unique in {split}")
                previous_id = document_id
                source_counts[source] += 1
                rows += 1
        expected = summary["split_summary"]["splits"][split]
        if rows != expected["documents"]:
            raise RuntimeError(f"{split} row count {rows} != preparation summary {expected['documents']}")
        expected_sources = {
            source: values["documents"] for source, values in expected["sources"].items()
        }
        if dict(source_counts) != expected_sources:
            raise RuntimeError(f"{split} source counts differ from the preparation summary")
        measured[split] = {"documents": rows, "sources": dict(sorted(source_counts.items()))}
        LOGGER.info("stage=manifest_check split=%s documents=%d", split, rows)
    return measured, paths


def cross_split_field_check(paths: dict[str, Path], field: str) -> dict:
    validation = set(iter_column(paths["validation"], field))
    test = set(iter_column(paths["test"], field))
    validation_rows = pq.ParquetFile(paths["validation"]).metadata.num_rows
    test_rows = pq.ParquetFile(paths["test"]).metadata.num_rows
    if len(validation) != validation_rows or len(test) != test_rows:
        raise RuntimeError(f"Duplicate {field} values inside validation or test")
    validation_test = len(validation & test)
    train_validation = 0
    train_test = 0
    for value in iter_column(paths["train"], field):
        train_validation += value in validation
        train_test += value in test
    if validation_test or train_validation or train_test:
        raise RuntimeError(
            f"Cross-split {field} leakage: train/validation={train_validation}, "
            f"train/test={train_test}, validation/test={validation_test}"
        )
    return {
        "train_validation": train_validation,
        "train_test": train_test,
        "validation_test": validation_test,
    }


def near_edge_check(connection: sqlite3.Connection, paths: dict[str, Path]) -> dict:
    cluster_ids = {
        row[0] for row in connection.execute("SELECT DISTINCT cluster_id FROM near_members")
    }
    cluster_split = {}
    for split, path in paths.items():
        for cluster_id in iter_column(path, "duplicate_cluster_id"):
            if cluster_id in cluster_ids:
                if cluster_id in cluster_split:
                    raise RuntimeError(f"Near cluster {cluster_id} has multiple retained manifest rows")
                cluster_split[cluster_id] = split
    if set(cluster_split) != cluster_ids:
        raise RuntimeError("One or more accepted near clusters have no retained manifest representative")
    mismatched_clusters = connection.execute(
        """
        SELECT COUNT(*) FROM near_edges e
        JOIN near_members l ON l.rowid = e.left_rowid
        JOIN near_members r ON r.rowid = e.right_rowid
        WHERE l.cluster_id <> r.cluster_id
        """
    ).fetchone()[0]
    if mismatched_clusters:
        raise RuntimeError(f"{mismatched_clusters} accepted near edges cross persisted clusters")
    accepted_edges = connection.execute("SELECT COUNT(*) FROM near_edges").fetchone()[0]
    return {
        "accepted_edges": accepted_edges,
        "persisted_clusters": len(cluster_ids),
        "edges_crossing_clusters": mismatched_clusters,
        "clusters_crossing_splits": 0,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    metadata = config.path("metadata")
    summary_path = metadata / "preparation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current_config_hash = canonical_json_hash(config.values)
    if summary["config_sha256"] != current_config_hash:
        raise RuntimeError("Current data pipeline config hash differs from the completed preparation config")

    database = config.path("interim") / "corpus_index.sqlite"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        exact = exact_report(metadata, connection)
        near = json.loads((metadata / "near_duplicate_report.json").read_text(encoding="utf-8"))
        raw = sum(values["raw"] for values in summary["accounting"]["sources"].values())
        empty = sum(values["removed_empty"] for values in summary["accounting"]["sources"].values())
        short = sum(values["removed_short"] for values in summary["accounting"]["sources"].values())
        exact_unique = summary["accounting"]["exact_unique_documents"]
        retained = exact_unique - near["removed_documents"]
        if raw != empty + short + exact["removed_documents"] + exact_unique:
            raise RuntimeError("Raw-to-exact corpus accounting does not reconcile")

        manifests, paths = validate_manifest_structure(config, summary)
        manifest_total = sum(item["documents"] for item in manifests.values())
        if manifest_total != retained:
            raise RuntimeError(f"Manifest total {manifest_total} != retained documents {retained}")

        field_checks = {
            field: cross_split_field_check(paths, field)
            for field in ("document_id", "canonical_text_hash", "duplicate_cluster_id")
        }
        edge_check = near_edge_check(connection, paths)
        if edge_check["accepted_edges"] != near["accepted_edges"]:
            raise RuntimeError("Near-edge report differs from the persisted database")
    finally:
        connection.close()

    expected_hashes = json.loads((metadata / "manifest_hashes.json").read_text(encoding="utf-8"))
    measured_hashes = {
        f"data/manifests/{path.name}": sha256_file(path) for path in paths.values()
    }
    if measured_hashes != expected_hashes:
        raise RuntimeError("One or more frozen manifest hashes changed")

    accounting = {
        "raw_core_documents": raw,
        "removed_empty": empty,
        "removed_short": short,
        "removed_invalid_or_too_short": empty + short,
        "exact_duplicate_removals": exact["removed_documents"],
        "exact_unique_documents": exact_unique,
        "near_candidate_pairs": near["candidate_pairs_checked"],
        "near_accepted_edges": near["accepted_edges"],
        "near_clusters": near["clusters"],
        "near_clustered_documents": near["clustered_documents"],
        "near_duplicate_removals": near["removed_documents"],
        "final_retained_documents": retained,
        "reconciliation": "raw = empty + short + exact removals + exact unique; retained = exact unique - near removals",
    }
    leakage = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "split_seed": config.values["seeds"]["split"],
        "split_policy": "deterministic document/cluster-level 90/5/5",
        "accounting": accounting,
        "manifests": manifests,
        "manifest_hashes": measured_hashes,
        "cross_split_intersections": field_checks,
        "near_edge_audit": edge_check,
        "exact_group_audit": {
            "groups": exact["groups"],
            "cross_source_groups": exact["cross_source_groups"],
            "policy": "Exact duplicates were removed globally before split assignment; canonical hashes are unique in retained manifests.",
        },
    }
    atomic_write_json(metadata / "exact_duplicate_report.json", exact)
    summary["exact_duplicates"] = exact
    summary["corpus_accounting"] = accounting
    atomic_write_json(summary_path, summary)
    atomic_write_json(metadata / "leakage_audit.json", leakage)
    LOGGER.info("stage=leakage_audit status=pass retained=%d", retained)


if __name__ == "__main__":
    main()
