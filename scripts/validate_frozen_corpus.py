"""Run the final real-artifact checks for the data pipeline handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, sha256_file
from src.data.paths import repository_relative, resolve_repository_path


HEX_ID = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen data pipeline manifests, tokenizer, counts, and subset.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rank(seed: int, document_id: str, label: str) -> str:
    return hashlib.sha256(f"{label}\0{seed}\0{document_id}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, ROOT)
    metadata = config.path("metadata")
    manifests = config.path("manifests")
    processed = config.path("processed")
    tokenizer_dir = config.path("tokenizer")

    prep = load_json(metadata / "preparation_summary.json")
    leakage = load_json(metadata / "leakage_audit.json")
    manifest_hashes = load_json(metadata / "manifest_hashes.json")
    token_report = load_json(metadata / "token_counts_by_source_split.json")
    tokenizer_audit = load_json(metadata / "tokenizer_audit.json")
    subset_summary = load_json(metadata / "training_subset_summary.json")
    raw_immutability = load_json(metadata / "raw_immutability.json")

    expected_sources = {
        source for source, values in config.values["sources"].items() if values["included_in_core"]
    }
    manifest_hash_check = {}
    for relative_path, expected_hash in manifest_hashes.items():
        actual_hash = sha256_file(ROOT / relative_path)
        manifest_hash_check[relative_path] = actual_hash
        if actual_hash != expected_hash:
            raise RuntimeError(f"Manifest hash changed: {relative_path}")

    index_uri = (config.path("interim") / "corpus_index.sqlite").resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(index_uri, uri=True)
    try:
        database_counts = {
            "exact_unique_documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "exact_duplicate_removals": connection.execute("SELECT COUNT(*) FROM exact_removed").fetchone()[0],
            "near_candidate_pairs": connection.execute("SELECT COUNT(*) FROM near_candidates").fetchone()[0],
            "near_accepted_edges": connection.execute("SELECT COUNT(*) FROM near_edges").fetchone()[0],
            "near_clustered_documents": connection.execute("SELECT COUNT(*) FROM near_members").fetchone()[0],
            "near_clusters": connection.execute("SELECT COUNT(DISTINCT cluster_id) FROM near_members").fetchone()[0],
            "near_duplicate_removals": connection.execute(
                "SELECT COUNT(*) FROM near_members WHERE removed = 1"
            ).fetchone()[0],
        }
    finally:
        connection.close()
    for name, measured in database_counts.items():
        if measured != prep["corpus_accounting"][name]:
            raise RuntimeError(f"Preparation count mismatch for {name}: {measured}")

    band_files = sorted((config.path("interim") / "near_bands").glob("band-*.bin"))
    band_sizes = [path.stat().st_size for path in band_files]
    if len(band_files) != config.values["near_duplicate"]["bands"] or len(set(band_sizes)) != 1:
        raise RuntimeError("Near-duplicate band files are incomplete or inconsistent")

    tokenizer_hashes = load_json(tokenizer_dir / "tokenizer_hashes.json")
    for filename, expected_hash in tokenizer_hashes.items():
        if sha256_file(tokenizer_dir / filename) != expected_hash:
            raise RuntimeError(f"Frozen tokenizer artifact changed: {filename}")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
    examples = ["Azərbaycan", "şəhərlərimizdəki", "öyrənəcəyik", "müəllimlərimizlə"]
    example_checks = []
    for text in examples:
        ids = processor.encode(text, out_type=int)
        decoded = processor.decode(ids)
        example_checks.append({"text": text, "ids": ids, "decoded": decoded})
        if ids != processor.encode(text, out_type=int) or decoded != text:
            raise RuntimeError(f"Tokenizer round-trip failed for {text}")
    special_tokens = load_json(tokenizer_dir / "special_tokens.json")
    if processor.vocab_size() != 16_000 or processor.unk_id() != 0 or processor.eos_id() != 1:
        raise RuntimeError("Frozen tokenizer vocabulary or special token IDs are invalid")
    if special_tokens["eod"]["piece"] != "<eod>" or processor.id_to_piece(1) != "<eod>":
        raise RuntimeError("The frozen EOD token is inconsistent")

    training_sample = pq.ParquetFile(tokenizer_dir / "training_sample_manifest.parquet")
    sample_documents = 0
    previous_id = None
    for batch in training_sample.iter_batches(batch_size=20_000):
        ids, _, _, splits, orders = [column.to_pylist() for column in batch.columns]
        for document_id, split, order in zip(ids, splits, orders):
            if split != "train" or order != sample_documents or not HEX_ID.fullmatch(document_id):
                raise RuntimeError("Tokenizer training sample is not a stable train-only sequence")
            if previous_id is not None and document_id <= previous_id:
                raise RuntimeError("Tokenizer training sample document IDs are not strictly sorted")
            previous_id = document_id
            sample_documents += 1
    if sample_documents != tokenizer_audit["training_provenance"]["documents"]:
        raise RuntimeError("Tokenizer training sample count changed")

    token_counts_path = metadata / "document_token_counts.parquet"
    token_parquet = pq.ParquetFile(token_counts_path)
    measured_counts: dict[str, dict[str, Counter]] = {}
    previous_by_split: dict[str, str] = {}
    for batch in token_parquet.iter_batches(batch_size=20_000):
        document_ids, sources, _, splits, counts, eod_flags = [column.to_pylist() for column in batch.columns]
        for document_id, source, split, token_count, includes_eod in zip(
            document_ids, sources, splits, counts, eod_flags
        ):
            if source not in expected_sources or split not in {"train", "validation", "test"}:
                raise RuntimeError("Token-count report contains an invalid source or split")
            if not HEX_ID.fullmatch(document_id) or token_count < 2 or not includes_eod:
                raise RuntimeError("Token-count report contains a malformed record")
            if split in previous_by_split and document_id <= previous_by_split[split]:
                raise RuntimeError(f"Token-count IDs are not strictly ordered within {split}")
            previous_by_split[split] = document_id
            measured_counts.setdefault(split, {}).setdefault(source, Counter()).update(
                {"documents": 1, "tokens": token_count}
            )
    for split, sources in token_report.items():
        for source, expected in sources.items():
            measured = measured_counts[split][source]
            if measured["documents"] != expected["documents"] or measured["tokens"] != expected["tokens"]:
                raise RuntimeError(f"Token totals changed for {split}/{source}")

    subset_path = manifests / "train_50m.parquet"
    subset_rows = pq.read_table(subset_path).to_pylist()
    subset_by_id = {}
    previous_order_key = None
    train_path = (processed / "train.parquet").resolve()
    for expected_order, row in enumerate(subset_rows):
        if row["sampling_order"] != expected_order or row["document_id"] in subset_by_id:
            raise RuntimeError("Training subset order or uniqueness is invalid")
        if resolve_repository_path(row["processed_file"], ROOT) != train_path or row["processed_row"] < 0:
            raise RuntimeError("Training subset contains an invalid processed reference")
        order_key = (rank(config.values["seeds"]["data"], row["document_id"], "order"), row["document_id"])
        if previous_order_key is not None and order_key <= previous_order_key:
            raise RuntimeError("Training subset does not follow the frozen data order")
        previous_order_key = order_key
        subset_by_id[row["document_id"]] = row
    matched_counts = 0
    for batch in token_parquet.iter_batches(batch_size=20_000, columns=["document_id", "split", "token_count"]):
        document_ids, splits, counts = [column.to_pylist() for column in batch.columns]
        for document_id, split, token_count in zip(document_ids, splits, counts):
            selected = subset_by_id.get(document_id)
            if selected is not None:
                if split != "train" or token_count != selected["token_count"]:
                    raise RuntimeError("Training subset token count or split mismatch")
                matched_counts += 1
    if matched_counts != len(subset_rows):
        raise RuntimeError("Training subset references missing token-count records")

    matched_references = 0
    row_number = 0
    train_parquet = pq.ParquetFile(train_path)
    for batch in train_parquet.iter_batches(batch_size=20_000, columns=["document_id"]):
        for document_id in batch.column(0).to_pylist():
            selected = subset_by_id.get(document_id)
            if selected is not None:
                if selected["processed_row"] != row_number:
                    raise RuntimeError("Training subset processed-row reference is stale")
                matched_references += 1
            row_number += 1
    if matched_references != len(subset_rows):
        raise RuntimeError("Training subset has unresolved processed references")

    selected_tokens = sum(row["token_count"] for row in subset_rows)
    if (
        len(subset_rows) != subset_summary["selected_documents"]
        or selected_tokens != subset_summary["selected_unique_tokens"]
        or sha256_file(subset_path) != subset_summary["manifest_sha256"]
        or selected_tokens < subset_summary["target_tokens"]
    ):
        raise RuntimeError("Training subset summary is stale")
    if leakage["status"] != "pass" or raw_immutability["status"] != "pass":
        raise RuntimeError("A prerequisite data pipeline safety gate is not passing")

    split_totals = {
        split: {
            "documents": sum(values["documents"] for values in sources.values()),
            "tokens": sum(values["tokens"] for values in sources.values()),
            "unknown_tokens": sum(values["unknown_tokens"] for values in sources.values()),
        }
        for split, sources in token_report.items()
    }
    result = {
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_state": {
            "stage": "prepare_complete",
            "evidence": "SQLite counts, final reports, retained manifests, and frozen hashes reconcile.",
            "database_counts": database_counts,
            "near_band_files": len(band_files),
            "near_band_bytes_each": band_sizes[0],
        },
        "manifest_hashes": manifest_hash_check,
        "leakage_status": leakage["status"],
        "tokenizer": {
            "vocab_size": processor.vocab_size(),
            "unk_id": processor.unk_id(),
            "eod_id": processor.eos_id(),
            "bos_id": processor.bos_id(),
            "pad_id": processor.pad_id(),
            "artifact_hashes": tokenizer_hashes,
            "training_sample_documents": sample_documents,
            "training_sample_split": "train",
            "round_trip_checks": example_checks,
        },
        "token_counts": {
            "path": repository_relative(token_counts_path, ROOT),
            "sha256": sha256_file(token_counts_path),
            "splits": split_totals,
            "total_documents": sum(values["documents"] for values in split_totals.values()),
            "total_tokens": sum(values["tokens"] for values in split_totals.values()),
        },
        "training_subset": {
            "path": repository_relative(subset_path, ROOT),
            "sha256": subset_summary["manifest_sha256"],
            "documents": len(subset_rows),
            "unique_tokens": selected_tokens,
            "target_consumed_tokens": subset_summary["target_tokens"],
            "matched_train_token_records": matched_counts,
            "matched_processed_references": matched_references,
            "duplicate_document_ids": 0,
            "validation_or_test_documents": 0,
            "streamable_reference_check": "pass",
        },
        "raw_immutability_status": raw_immutability["status"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "pyarrow",
                    "numpy",
                    "sentencepiece",
                    "protobuf",
                    "datasketch",
                    "PyYAML",
                    "pytest",
                    "matplotlib",
                )
            },
        },
    }
    atomic_write_json(metadata / "frozen_corpus_validation.json", result)
    print(json.dumps({"status": result["status"], "token_counts": result["token_counts"], "training_subset": result["training_subset"]}, indent=2))


if __name__ == "__main__":
    main()
