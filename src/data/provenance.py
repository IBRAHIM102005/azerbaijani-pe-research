"""Build the local source registry from verified metadata."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import yaml

from .config import DataPipelineConfig
from .hashing import atomic_write_text, sha256_file


def build_source_registry(
    config: DataPipelineConfig, inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record local source facts without filling unknown provenance fields."""

    readme = config.path("original_dollma") / "README.md"
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for shard in inventory:
        hashes[shard["source"]].append(
            {"filename": shard["filename"], "sha256": shard["sha256"]}
        )

    components = {}
    for source, settings in config.values["sources"].items():
        components[source] = {
            "raw_folder": source,
            "group": settings["group"],
            "component_label": settings.get("component_label"),
            "included_in_core": settings["included_in_core"],
            "status": settings.get("status", "included_core"),
            "evidence": settings["evidence"],
            "text_column": settings["text_column"],
            "source_level_license": None,
            "source_level_revision": None,
            "raw_shards": hashes.get(source, []),
        }
    registry = {
        "dataset": "DOLLMA",
        "dataset_declared_license": "cc-by-nc-sa-4.0",
        "local_access_date": date.today().isoformat(),
        "access_date_basis": "Date the local source inventory was generated and verified.",
        "license_evidence": str(readme.resolve()),
        "license_evidence_path_role": "external provenance display path; runtime root is config-resolved",
        "license_evidence_sha256": sha256_file(readme),
        "dataset_revision": None,
        "revision_note": "Not available from the local README; Git metadata was not inspected in this no-Git task.",
        "components": components,
    }
    output = yaml.safe_dump(registry, allow_unicode=True, sort_keys=False)
    atomic_write_text(config.path("metadata") / "source_registry.yaml", output)
    return registry
