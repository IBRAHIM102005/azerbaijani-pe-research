"""Repair near-duplicate state without repeating valid upstream preparation."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, canonical_json_hash, sha256_file
from src.data.manifests import write_processed_and_manifests
from src.data.near import build_candidate_table, build_clusters, verify_candidate_edges


LOGGER = logging.getLogger("data_pipeline.repair_near")
EXPECTED_UPSTREAM = {
    "documents": 6_209_184,
    "exact_removed": 144_480,
    "removals": 1_873_990,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair data pipeline from frozen near-band files onward.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    parser.add_argument(
        "--through",
        choices=("candidates", "verify", "clusters", "manifests"),
        default="manifests",
        help="Stop after this durable repair stage.",
    )
    return parser.parse_args()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _validate_upstream(database: Path, band_paths: list[Path]) -> dict:
    uri = "file:" + database.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in EXPECTED_UPSTREAM
        }
    finally:
        connection.close()
    if quick_check != "ok" or counts != EXPECTED_UPSTREAM:
        raise RuntimeError(f"Upstream data pipeline state failed validation: check={quick_check} counts={counts}")
    expected_band_bytes = counts["documents"] * 24
    band_hashes = {}
    for path in band_paths:
        if not path.is_file() or path.stat().st_size != expected_band_bytes:
            raise RuntimeError(f"Invalid frozen band file: {path}")
        band_hashes[path.name] = sha256_file(path)
    return {
        "database_quick_check": quick_check,
        "table_counts": counts,
        "expected_band_bytes": expected_band_bytes,
        "band_hashes": band_hashes,
    }


def _prepare_staging_database(source: Path, staging: Path, upstream: dict) -> None:
    if staging.exists():
        staged = _validate_upstream(staging, [])
        if staged["table_counts"] != upstream["table_counts"]:
            raise RuntimeError("Existing staged database does not match frozen upstream counts")
        return
    wal = source.with_name(source.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Source database has an active WAL; refuse an unsafe file copy")
    temporary = staging.with_name(staging.name + ".copying")
    if temporary.exists():
        temporary.unlink()
    LOGGER.info("stage=repair_copy_start bytes=%d output=%s", source.stat().st_size, staging)
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    shutil.copystat(source, temporary)
    os.replace(temporary, staging)
    copied = _validate_upstream(staging, [Path(name) for name in []])
    if copied["table_counts"] != upstream["table_counts"]:
        raise RuntimeError("Staged database upstream counts differ after copy")


def _initialize_repair_tables(connection: sqlite3.Connection) -> None:
    tables = _table_names(connection)
    for name in ("near_candidates", "near_edges", "near_members"):
        prerepair = f"{name}_prerepair"
        if name in tables and prerepair not in tables:
            connection.execute(f"ALTER TABLE {name} RENAME TO {prerepair}")
            tables.remove(name)
            tables.add(prerepair)
    for name in (
        "near_candidate_progress",
        "near_verification_progress",
        "m1_stage_state",
    ):
        if name in tables:
            connection.execute(f"DROP TABLE {name}")
    connection.commit()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    interim = config.path("interim")
    source_database = interim / "corpus_index.sqlite"
    staging_database = interim / "corpus_index_repair.sqlite"
    settings = config.values["near_duplicate"]
    band_paths = [
        interim / "near_bands" / f"band-{index:02d}.bin" for index in range(settings["bands"])
    ]
    started = time.perf_counter()
    upstream = _validate_upstream(source_database, band_paths)
    _prepare_staging_database(source_database, staging_database, upstream)

    connection = sqlite3.connect(staging_database)
    connection.executescript(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;"
    )
    try:
        tables = _table_names(connection)
        if "near_candidates_prerepair" not in tables:
            _initialize_repair_tables(connection)
        candidates = build_candidate_table(
            connection,
            band_paths,
            max_bucket_size=settings["max_complete_band_bucket"],
        )
        if candidates["skipped"]:
            raise RuntimeError(f"Corrected candidate generation skipped {candidates['skipped']} buckets")
        if args.through == "candidates":
            near = {}
            split_summary = {}
            manifest_hashes = {}
        else:
            verified = verify_candidate_edges(
                connection,
                threshold=settings["selected_threshold"],
                shingle_size=settings["shingle_size"],
            )
            near = {**verified}
            if args.through == "verify":
                split_summary = {}
                manifest_hashes = {}
            else:
                near.update(build_clusters(connection))
                if args.through == "clusters":
                    split_summary = {}
                    manifest_hashes = {}
                else:
                    processed_staging = config.path("processed").with_name("corpus_repair")
                    manifests_staging = config.path("manifests").with_name("manifests_repair")
                    split_summary, manifest_hashes = write_processed_and_manifests(
                        connection,
                        processed_staging,
                        manifests_staging,
                        config.values["sources"],
                        config.values["split"],
                        config.values["seeds"]["split"],
                        repo_root=config.repo_root,
                        reference_processed_dir=config.path("processed"),
                    )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
        stage_states = {
            row[0]: {"version": row[1], "config_hash": row[2], "status": row[3]}
            for row in connection.execute(
                "SELECT stage, version, config_hash, status FROM m1_stage_state ORDER BY stage"
            )
        }
    finally:
        connection.close()

    result = {
        "repair_version": "near-repair-v2",
        "config_canonical_sha256": canonical_json_hash(config.values),
        "upstream": upstream,
        "staging_database": staging_database.name,
        "candidate_generation": candidates,
        "near_duplicates": near,
        "split_summary": split_summary,
        "manifest_hashes": manifest_hashes,
        "stage_states": stage_states,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "promotion_status": "not_promoted_pending_independent_validation",
    }
    atomic_write_json(config.path("metadata") / "near_repair_stage_summary.json", result)
    LOGGER.info(
        "stage=repair_checkpoint through=%s candidates=%d accepted=%s output=%s",
        args.through,
        candidates["unique_candidate_pairs"],
        near.get("accepted_edges"),
        staging_database,
    )


if __name__ == "__main__":
    main()
