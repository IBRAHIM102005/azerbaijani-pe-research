import json

from src.data.hashing import (
    atomic_write_json,
    canonical_text_hash,
    document_id,
    raw_record_id,
)


def test_hashes_are_stable_and_source_aware():
    text = "Azərbaycan dili"
    assert canonical_text_hash(text) == canonical_text_hash(text)
    assert document_id("azwiki", text) == document_id("azwiki", text)
    assert document_id("azwiki", text) != document_id("anl-news", text)


def test_raw_record_id_changes_with_provenance():
    assert raw_record_id("azwiki", "part.parquet", 1) != raw_record_id(
        "azwiki", "part.parquet", 2
    )


def test_atomic_json_replaces_complete_content_without_partial_file(tmp_path):
    output = tmp_path / "metadata.json"
    output.write_text('{"status":"old"}\n', encoding="utf-8")

    atomic_write_json(output, {"status": "complete", "records": 42})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "records": 42,
        "status": "complete",
    }
    assert list(tmp_path.glob(f".{output.name}.*")) == []
