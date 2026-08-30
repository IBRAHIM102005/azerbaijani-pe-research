"""Verify original and local DOLLMA shards against the frozen inventory."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hashing import atomic_write_json, sha256_file
from src.data.config import load_config
from src.data.paths import repository_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash raw DOLLMA shards and compare them with the data pipeline baseline.")
    parser.add_argument("--inventory", type=Path, default=ROOT / "data" / "metadata" / "raw_inventory.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "metadata" / "raw_immutability.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    checks = []
    mismatches = []
    for index, shard in enumerate(inventory, start=1):
        targets = [
            (
                "original",
                config.path("original_dollma") / shard["source"] / shard["filename"],
                shard["sha256"],
                shard["bytes"],
            )
        ]
        if shard.get("local_core_path"):
            targets.append(
                (
                    "local_core_copy",
                    config.path("raw_core") / shard["source"] / shard["filename"],
                    shard["local_core_sha256"],
                    shard["bytes"],
                )
            )
        for location, path, expected_hash, expected_bytes in targets:
            exists = path.is_file()
            actual_bytes = path.stat().st_size if exists else None
            actual_hash = sha256_file(path) if exists else None
            passed = exists and actual_bytes == expected_bytes and actual_hash == expected_hash
            record = {
                "source": shard["source"],
                "filename": shard["filename"],
                "location": location,
                "path": (
                    repository_relative(path, ROOT)
                    if location == "local_core_copy"
                    else str(path.resolve())
                ),
                "path_role": (
                    "repository_relative_local_copy"
                    if location == "local_core_copy"
                    else "external_provenance_runtime_root"
                ),
                "exists": exists,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": "pass" if passed else "fail",
            }
            checks.append(record)
            if not passed:
                mismatches.append(record)
        logging.info("stage=raw_hash_progress shards=%d/%d source=%s", index, len(inventory), shard["source"])
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_path": repository_relative(args.inventory, ROOT),
        "external_root_resolution": "AZ_PE_DOLLMA_ROOT or the config-relative original_dollma path",
        "original_shards_checked": len(inventory),
        "local_core_shards_checked": sum(bool(shard.get("local_core_path")) for shard in inventory),
        "checks": checks,
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
    }
    atomic_write_json(args.output, result)
    if mismatches:
        raise RuntimeError(f"Raw immutability failed for {len(mismatches)} path(s)")
    logging.info("stage=raw_immutability status=pass checks=%d", len(checks))


if __name__ == "__main__":
    main()
