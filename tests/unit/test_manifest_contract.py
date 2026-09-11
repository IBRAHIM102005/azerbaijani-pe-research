from src.data.manifests import MANIFEST_SCHEMA, PROCESSED_SCHEMA


def test_processed_corpus_schema_keeps_training_and_provenance_fields():
    assert PROCESSED_SCHEMA.names == [
        "document_id",
        "source",
        "source_group",
        "text",
        "character_count",
        "approximate_word_count",
        "split",
        "duplicate_cluster_id",
        "canonical_text_hash",
        "raw_record_id",
        "raw_shard",
        "raw_row_index",
        "quality_flags",
    ]


def test_manifest_schema_is_compact_but_traceable():
    assert MANIFEST_SCHEMA.names == [
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
    ]

    assert "text" not in MANIFEST_SCHEMA.names
    assert {"processed_file", "processed_row"} <= set(MANIFEST_SCHEMA.names)
    assert {"raw_record_id", "raw_shard", "raw_row_index"} <= set(
        MANIFEST_SCHEMA.names
    )
