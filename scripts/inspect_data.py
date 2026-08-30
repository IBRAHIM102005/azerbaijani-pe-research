"""Inspect raw DOLLMA shards and profile the core sources."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.dedup import run_near_duplicate_pilot
from src.data.hashing import atomic_write_json, canonical_json_hash
from src.data.io import source_shards, stream_source
from src.data.normalize import normalize_text, unicode_letter_count
from src.data.provenance import build_source_registry
from src.data.quality import SourceProfile
from src.data.schema import inspect_parquet


LOGGER = logging.getLogger("data_pipeline.inspect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DOLLMA schemas and profile the data pipeline core corpus.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def write_inventory_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "source", "filename", "path", "bytes", "sha256", "rows", "row_groups",
        "schema_signature", "selected_text_column", "null_text_count", "included_in_core",
        "local_core_path", "local_core_sha256", "local_copy_matches_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def inspect_inventory(config) -> list[dict]:
    original_root = config.path("original_dollma")
    raw_core = config.path("raw_core")
    if not original_root.is_dir():
        raise FileNotFoundError(f"Original DOLLMA directory is missing: {original_root}")
    if not raw_core.is_dir():
        raise FileNotFoundError(f"Local core DOLLMA directory is missing: {raw_core}")

    inventory = []
    for source, settings in config.values["sources"].items():
        for shard in source_shards(original_root, source):
            LOGGER.info("stage=inventory source=%s shard=%s", source, shard.name)
            record = inspect_parquet(shard, source, settings["text_column"])
            record["included_in_core"] = settings["included_in_core"]
            local_path = raw_core / source / shard.name
            if settings["included_in_core"]:
                if not local_path.is_file():
                    raise FileNotFoundError(f"Core shard is missing from the repository data path: {local_path}")
                local_hash = inspect_parquet(local_path, source, settings["text_column"])["sha256"]
                record["local_core_path"] = str(local_path.resolve())
                record["local_core_sha256"] = local_hash
                record["local_copy_matches_source"] = local_hash == record["sha256"]
                if not record["local_copy_matches_source"]:
                    raise RuntimeError(f"Local raw copy differs from the original shard: {local_path}")
            else:
                record["local_core_path"] = None
                record["local_core_sha256"] = None
                record["local_copy_matches_source"] = None
            inventory.append(record)
    return inventory


def profile_core(config, batch_size: int) -> tuple[dict, dict]:
    raw_root = config.path("raw_core")
    pilot_settings = config.values["near_duplicate"]
    profiles = {}
    pilot_documents = []
    started = time.perf_counter()
    for source in config.included_sources:
        source_started = time.perf_counter()
        settings = config.source(source)
        profile = SourceProfile(source, pilot_settings["pilot_documents_per_source"], config.values["seeds"]["data"])
        for record in stream_source(raw_root, source, settings["text_column"], batch_size):
            profile.update(record.shard, record.row_index, record.text)
        profiles[source] = profile.finalize()
        for document in profile.sample.documents():
            normalized = normalize_text(document.text).text
            if unicode_letter_count(normalized) >= config.values["normalization"]["minimum_unicode_letters"]:
                pilot_documents.append(type(document)(document.source, document.record_id, normalized))
        LOGGER.info(
            "stage=profile source=%s records=%d runtime_seconds=%.3f",
            source,
            profile.documents,
            time.perf_counter() - source_started,
        )

    global_summary = {
        "documents": sum(item["documents"] for item in profiles.values()),
        "null_documents": sum(item["null_documents"] for item in profiles.values()),
        "empty_documents": sum(item["empty_documents"] for item in profiles.values()),
        "whitespace_only_documents": sum(item["whitespace_only_documents"] for item in profiles.values()),
        "total_characters": sum(item["total_characters"] for item in profiles.values()),
        "approximate_whitespace_words": sum(item["approximate_whitespace_words"] for item in profiles.values()),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    pilot = run_near_duplicate_pilot(
        pilot_documents,
        shingle_size=pilot_settings["shingle_size"],
        fingerprint_size=pilot_settings["fingerprint_size"],
        bands=pilot_settings["bands"],
        thresholds=pilot_settings["candidate_thresholds"],
    )
    pilot["selection_method"] = "Smallest stable provenance hashes per source after normalization and the 50-letter candidate filter."
    return {"global": global_summary, "sources": profiles}, pilot


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    metadata = config.path("metadata")
    metadata.mkdir(parents=True, exist_ok=True)

    inventory = inspect_inventory(config)
    atomic_write_json(metadata / "raw_inventory.json", inventory)
    write_inventory_csv(metadata / "raw_inventory.csv", inventory)
    build_source_registry(config, inventory)
    atomic_write_json(
        metadata / "raw_inventory_hash.json",
        {
            "raw_inventory_sha256": canonical_json_hash(inventory),
            "shards": len(inventory),
            "rows": sum(item["rows"] for item in inventory),
            "bytes": sum(item["bytes"] for item in inventory),
        },
    )

    profile, pilot = profile_core(config, args.batch_size)
    atomic_write_json(metadata / "raw_quality_profile.json", profile)
    atomic_write_json(metadata / "near_duplicate_pilot.json", pilot)
    LOGGER.info(
        "stage=inspect_complete shards=%d core_records=%d output=%s",
        len(inventory),
        profile["global"]["documents"],
        metadata,
    )


if __name__ == "__main__":
    main()
