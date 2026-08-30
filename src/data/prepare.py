"""Build the canonical, deduplicated, split data pipeline corpus."""

from __future__ import annotations

import csv
import logging
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DataPipelineConfig
from .hashing import canonical_text_hash, document_id, raw_record_id, sha256_file
from .io import stream_source
from .manifests import write_processed_and_manifests
from .near import build_band_files, build_candidate_table, verify_and_cluster
from .normalize import normalize_text, unicode_letter_count
from .quality import quality_flags


LOGGER = logging.getLogger("data_pipeline.prepare")


DOCUMENT_SCHEMA = """
CREATE TABLE documents (
    record_key INTEGER PRIMARY KEY,
    canonical_hash TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    raw_record_id TEXT NOT NULL,
    source TEXT NOT NULL,
    shard TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    letter_count INTEGER NOT NULL,
    quality_flags TEXT NOT NULL,
    exact_group_size INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE exact_removed (
    raw_record_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    shard TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    canonical_hash TEXT NOT NULL,
    representative_document_id TEXT NOT NULL,
    representative_source TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE removals (
    raw_record_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    shard TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    reason TEXT NOT NULL,
    letter_count INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TRIGGER capture_exact_duplicate BEFORE INSERT ON documents
WHEN EXISTS (SELECT 1 FROM documents WHERE canonical_hash = NEW.canonical_hash)
BEGIN
    INSERT INTO exact_removed
    SELECT NEW.raw_record_id, NEW.source, NEW.shard, NEW.row_index, NEW.document_id,
           NEW.canonical_hash, document_id, source
    FROM documents WHERE canonical_hash = NEW.canonical_hash;
    UPDATE documents SET exact_group_size = exact_group_size + 1
    WHERE canonical_hash = NEW.canonical_hash;
    SELECT RAISE(IGNORE);
END;
"""


def create_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;"
    )
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if "documents" not in tables:
        connection.executescript(DOCUMENT_SCHEMA)
    return connection


def ingest_core(
    config: DataPipelineConfig, connection: sqlite3.Connection, batch_size: int = 4096
) -> dict[str, Any]:
    """Normalize, filter, and exact-deduplicate all core records."""

    minimum_letters = config.values["normalization"]["minimum_unicode_letters"]
    accounting: dict[str, Counter[str]] = defaultdict(Counter)
    flag_counts: dict[str, Counter[str]] = defaultdict(Counter)
    insert_sql = """
        INSERT INTO documents (
            canonical_hash, document_id, raw_record_id, source, shard, row_index,
            text, character_count, word_count, letter_count, quality_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    removal_sql = "INSERT INTO removals VALUES (?, ?, ?, ?, ?, ?)"
    started = time.perf_counter()
    for source in config.included_sources:
        source_started = time.perf_counter()
        settings = config.source(source)
        inserts = []
        removals = []
        for record in stream_source(config.path("raw_core"), source, settings["text_column"], batch_size):
            counts = accounting[source]
            counts["raw"] += 1
            counts["parsed"] += 1
            record_id = raw_record_id(source, record.shard, record.row_index)
            normalized = normalize_text(record.text)
            counts["normalized"] += 1
            counts["changed_nfc"] += normalized.changed_nfc
            counts["changed_newlines"] += normalized.changed_newlines
            counts["changed_horizontal_space"] += normalized.changed_horizontal_space
            counts["removed_control_characters"] += normalized.removed_control_characters
            if not normalized.text:
                counts["removed_empty"] += 1
                removals.append((record_id, source, record.shard, record.row_index, "empty", 0))
            else:
                letters = unicode_letter_count(normalized.text)
                if letters < minimum_letters:
                    counts["removed_short"] += 1
                    removals.append((record_id, source, record.shard, record.row_index, "short", letters))
                else:
                    flags = sorted(quality_flags(normalized.text))
                    flag_counts[source].update(flags)
                    text_hash = canonical_text_hash(normalized.text)
                    inserts.append(
                        (
                            text_hash,
                            document_id(source, normalized.text),
                            record_id,
                            source,
                            record.shard,
                            record.row_index,
                            normalized.text,
                            len(normalized.text),
                            len(normalized.text.split()),
                            letters,
                            ",".join(flags),
                        )
                    )
            if len(inserts) + len(removals) >= batch_size:
                if inserts:
                    connection.executemany(insert_sql, inserts)
                    inserts.clear()
                if removals:
                    connection.executemany(removal_sql, removals)
                    removals.clear()
                connection.commit()
        if inserts:
            connection.executemany(insert_sql, inserts)
        if removals:
            connection.executemany(removal_sql, removals)
        connection.commit()
        LOGGER.info(
            "stage=normalize_exact source=%s records=%d runtime_seconds=%.3f",
            source,
            accounting[source]["raw"],
            time.perf_counter() - source_started,
        )

    exact_by_source = dict(
        connection.execute("SELECT source, COUNT(*) FROM exact_removed GROUP BY source")
    )
    for source in config.included_sources:
        accounting[source]["exact_duplicate"] = exact_by_source.get(source, 0)
    total_unique = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    return {
        "sources": {source: dict(counts) for source, counts in accounting.items()},
        "quality_flags_after_normalization": {source: dict(flags) for source, flags in flag_counts.items()},
        "exact_unique_documents": total_unique,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def write_exact_reports(connection: sqlite3.Connection, metadata_dir: Path) -> dict[str, Any]:
    """Export exact-duplicate group summaries and removed provenance."""

    group_path = metadata_dir / "exact_duplicate_groups.csv"
    member_path = metadata_dir / "exact_duplicate_members.csv"
    with group_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["canonical_hash", "group_size", "sources", "selected_representative"])
        for row in connection.execute(
            """
            SELECT d.canonical_hash, d.exact_group_size,
                   GROUP_CONCAT(DISTINCT x.source), d.document_id
            FROM documents d JOIN (
                SELECT canonical_hash, source FROM exact_removed
                UNION ALL SELECT canonical_hash, source FROM documents
            ) x ON x.canonical_hash = d.canonical_hash
            WHERE d.exact_group_size > 1
            GROUP BY d.canonical_hash, d.exact_group_size, d.document_id
            ORDER BY d.canonical_hash
            """
        ):
            writer.writerow(row)
    with member_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "canonical_hash", "removed_raw_record_id", "removed_source", "removed_shard",
            "removed_row_index", "removed_document_id", "selected_representative", "representative_source",
        ])
        for row in connection.execute(
            """SELECT canonical_hash, raw_record_id, source, shard, row_index, document_id,
                      representative_document_id, representative_source
               FROM exact_removed ORDER BY canonical_hash, raw_record_id"""
        ):
            writer.writerow(row)

    total_removed = connection.execute("SELECT COUNT(*) FROM exact_removed").fetchone()[0]
    groups = connection.execute("SELECT COUNT(*) FROM documents WHERE exact_group_size > 1").fetchone()[0]
    cross_groups = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT canonical_hash FROM (
                SELECT canonical_hash, source FROM documents
                UNION ALL
                SELECT canonical_hash, source FROM exact_removed
            )
            GROUP BY canonical_hash
            HAVING COUNT(DISTINCT source) > 1
        )
        """
    ).fetchone()[0]
    return {
        "groups": groups,
        "removed_documents": total_removed,
        "cross_source_groups": cross_groups,
        "group_report": str(group_path.resolve()),
        "group_report_bytes": group_path.stat().st_size,
        "group_report_sha256": sha256_file(group_path),
        "member_report": str(member_path.resolve()),
        "member_report_bytes": member_path.stat().st_size,
        "member_report_sha256": sha256_file(member_path),
    }


def run_prepare(config: DataPipelineConfig, index_path: Path) -> dict[str, Any]:
    connection = create_index(index_path)
    try:
        doc_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count > 0:
            LOGGER.info("Resuming: documents table already populated with %d rows. Reconstructing accounting...", doc_count)
            accounting = {"sources": {}, "quality_flags_after_normalization": {}}
            doc_counts = dict(connection.execute("SELECT source, COUNT(*) FROM documents GROUP BY source").fetchall())
            exact_counts = dict(connection.execute("SELECT source, COUNT(*) FROM exact_removed GROUP BY source").fetchall())
            empty_counts = dict(connection.execute("SELECT source, COUNT(*) FROM removals WHERE reason='empty' GROUP BY source").fetchall())
            short_counts = dict(connection.execute("SELECT source, COUNT(*) FROM removals WHERE reason='short' GROUP BY source").fetchall())
            
            for source in config.included_sources:
                docs = doc_counts.get(source, 0)
                exact_count = exact_counts.get(source, 0)
                empty = empty_counts.get(source, 0)
                short = short_counts.get(source, 0)
                raw = docs + exact_count + empty + short
                accounting["sources"][source] = {
                    "raw": raw, "parsed": raw, "normalized": raw,
                    "removed_empty": empty, "removed_short": short,
                    "exact_duplicate": exact_count,
                    "changed_nfc": "unknown_resumed",
                    "changed_newlines": "unknown_resumed",
                    "changed_horizontal_space": "unknown_resumed",
                    "removed_control_characters": "unknown_resumed",
                }
            accounting["exact_unique_documents"] = doc_count
            accounting["runtime_seconds"] = "unknown_resumed"
            
            # Skip write_exact_reports if we are resuming and files exist
            group_path = config.path("metadata") / "exact_duplicate_groups.csv"
            member_path = config.path("metadata") / "exact_duplicate_members.csv"
            if group_path.exists() and member_path.exists():
                LOGGER.info("Resuming: exact duplicate reports already exist. Skipping write_exact_reports...")
                exact = {
                    "groups": "unknown_resumed",
                    "removed_documents": "unknown_resumed",
                    "cross_source_groups": "unknown_resumed",
                    "group_report": str(group_path.resolve()),
                    "member_report": str(member_path.resolve()),
                }
            else:
                exact = write_exact_reports(connection, config.path("metadata"))
            
            # Clean up potentially incomplete near dedup tables
            connection.executescript(
                """
                DROP TABLE IF EXISTS near_candidates;
                DROP TABLE IF EXISTS near_edges;
                DROP TABLE IF EXISTS near_members;
                DROP TABLE IF EXISTS near_candidate_progress;
                DROP TABLE IF EXISTS near_verification_progress;
                DROP TABLE IF EXISTS m1_stage_state;
                """
            )
        else:
            accounting = ingest_core(config, connection)
            exact = write_exact_reports(connection, config.path("metadata"))
        
        settings = config.values["near_duplicate"]
        bands_dir = config.path("interim") / "near_bands"
        bands_expected = settings["bands"]
        band_paths_list = [bands_dir / f"band-{i:02d}.bin" for i in range(bands_expected)]
        band_files_valid = doc_count > 0 and all(p.exists() and p.stat().st_size == doc_count * 24 for p in band_paths_list)

        if band_files_valid:
            LOGGER.info("Resuming: near_bands are fully populated. Skipping build_band_files...")
            band_paths = band_paths_list
            fingerprinted = doc_count
        else:
            band_paths, fingerprinted = build_band_files(
                connection,
                bands_dir,
                shingle_size=settings["shingle_size"],
                fingerprint_size=settings["fingerprint_size"],
                bands=settings["bands"],
            )
        candidates = build_candidate_table(
            connection,
            band_paths,
            max_bucket_size=settings["max_complete_band_bucket"],
        )
        if candidates["skipped"]:
            raise RuntimeError(f"Near-duplicate candidate generation skipped {candidates['skipped']} oversized buckets")
        near = verify_and_cluster(
            connection,
            threshold=settings["selected_threshold"],
            shingle_size=settings["shingle_size"],
        )
        near["fingerprinted_documents"] = fingerprinted
        near["candidate_generation"] = candidates
        summary, manifest_hashes = write_processed_and_manifests(
            connection,
            config.path("processed"),
            config.path("manifests"),
            config.values["sources"],
            config.values["split"],
            config.values["seeds"]["split"],
            repo_root=config.repo_root,
        )
        return {
            "accounting": accounting,
            "exact_duplicates": exact,
            "near_duplicates": near,
            "split_summary": summary,
            "manifest_hashes": manifest_hashes,
        }
    finally:
        connection.close()
