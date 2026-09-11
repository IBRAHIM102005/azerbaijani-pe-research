from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _registry():
    return yaml.safe_load(
        (ROOT / "data/metadata/source_registry.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_core_contains_only_approved_native_components():
    components = _registry()["components"]

    included = {
        name
        for name, metadata in components.items()
        if metadata["included_in_core"]
    }

    assert included == {
        "anl-news",
        "azwiki",
        "elite-blogs",
        "elite-books",
        "eqanun",
        "mediocore-books",
    }

    assert sum(
        len(components[name]["raw_shards"])
        for name in included
    ) == 14


def test_translated_and_unresolved_sources_remain_excluded():
    components = _registry()["components"]

    translated = components["translated-enwiki"]
    assert translated["included_in_core"] is False
    assert translated["status"] == "excluded_translated_source"

    bhos = components["bhos"]
    assert bhos["included_in_core"] is False
    assert bhos["status"] == "requires_source_decision"


def test_source_registry_keeps_sha256_for_every_recorded_shard():
    components = _registry()["components"]

    for metadata in components.values():
        for shard in metadata["raw_shards"]:
            assert len(shard["sha256"]) == 64
