import sqlite3

import numpy as np

from src.data.near import (
    build_candidate_table,
    build_clusters,
    stage_is_complete,
    verify_candidate_edges,
)


BAND_DTYPE = np.dtype([("key", "S16"), ("rowid", "<u8")])


def _band(path, rowids, key=b"same-bucket-key"):
    records = np.empty(len(rowids), dtype=BAND_DTYPE)
    records["key"] = key
    records["rowid"] = rowids
    records.tofile(path)


def _candidate_rows(connection):
    return list(
        connection.execute(
            "SELECT left_rowid, right_rowid FROM near_candidates ORDER BY left_rowid, right_rowid"
        )
    )


def test_large_bucket_emits_non_anchor_pairs(tmp_path):
    band = tmp_path / "band.bin"
    _band(band, range(1, 204))
    connection = sqlite3.connect(":memory:")
    result = build_candidate_table(connection, [band], max_bucket_size=10_000)
    assert result["unique_candidate_pairs"] == 203 * 202 // 2
    assert result["star"] == 0
    assert result["skipped"] == 0
    assert connection.execute(
        "SELECT 1 FROM near_candidates WHERE left_rowid = 2 AND right_rowid = 3"
    ).fetchone()


def test_large_bucket_verifies_non_anchor_true_pair(tmp_path):
    band = tmp_path / "band.bin"
    _band(band, range(1, 204))
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE documents (document_id TEXT NOT NULL, text TEXT NOT NULL, character_count INTEGER NOT NULL)"
    )
    common = "Bu, uzun Azərbaycan sənədinin dəyişməyən hissəsidir. " * 40
    texts = ["Tamamilə ayrı lövbər mətni.", common + "B", common + "C"]
    texts.extend(f"Fərqli sintetik sənəd {index}." for index in range(4, 204))
    connection.executemany(
        "INSERT INTO documents VALUES (?, ?, ?)",
        [(f"doc-{index}", text, len(text)) for index, text in enumerate(texts, start=1)],
    )
    build_candidate_table(connection, [band], max_bucket_size=10_000)
    verify_candidate_edges(connection, threshold=0.95, shingle_size=5)
    assert connection.execute(
        "SELECT similarity FROM near_edges WHERE left_rowid = 2 AND right_rowid = 3"
    ).fetchone()[0] >= 0.95


def test_large_bucket_output_is_deterministic_and_canonical(tmp_path):
    band = tmp_path / "band.bin"
    _band(band, range(205, 0, -1))
    outputs = []
    for _ in range(2):
        connection = sqlite3.connect(":memory:")
        build_candidate_table(connection, [band], max_bucket_size=10_000)
        outputs.append(_candidate_rows(connection))
    assert outputs[0] == outputs[1]
    assert all(left < right for left, right in outputs[0])


def test_candidates_are_deduplicated_across_bands(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    _band(first, range(1, 12))
    _band(second, range(1, 12))
    connection = sqlite3.connect(":memory:")
    result = build_candidate_table(connection, [first, second], max_bucket_size=10_000)
    assert result["pair_insert_attempts"] == 110
    assert result["unique_candidate_pairs"] == 55


def test_candidate_resume_matches_clean_execution(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    _band(first, range(1, 15), b"first")
    _band(second, range(8, 23), b"second")
    resumed = sqlite3.connect(":memory:")
    build_candidate_table(resumed, [first], max_bucket_size=10_000)
    build_candidate_table(resumed, [first, second], max_bucket_size=10_000)
    clean = sqlite3.connect(":memory:")
    build_candidate_table(clean, [first, second], max_bucket_size=10_000)
    assert _candidate_rows(resumed) == _candidate_rows(clean)


def test_connected_components_are_transitive_and_deterministic():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE documents (document_id TEXT)")
    connection.executemany(
        "INSERT INTO documents(document_id) VALUES (?)", [("c",), ("a",), ("b",)]
    )
    connection.execute(
        "CREATE TABLE near_edges (left_rowid INTEGER, right_rowid INTEGER, similarity REAL)"
    )
    connection.executemany(
        "INSERT INTO near_edges VALUES (?, ?, ?)", [(1, 2, 0.96), (2, 3, 0.96)]
    )
    result = build_clusters(connection)
    rows = list(connection.execute("SELECT cluster_id, representative_rowid FROM near_members"))
    assert result == {"clusters": 1, "clustered_documents": 3, "removed_documents": 2}
    assert len({row[0] for row in rows}) == 1
    assert {row[1] for row in rows} == {2}


def test_incomplete_stage_is_not_accepted_as_complete():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE m1_stage_state (
               stage TEXT PRIMARY KEY, version TEXT, config_hash TEXT, status TEXT,
               details_json TEXT, updated_at_utc TEXT
           ) WITHOUT ROWID"""
    )
    connection.execute(
        "INSERT INTO m1_stage_state VALUES ('near_candidates','2','frozen','running','{}','now')"
    )
    assert not stage_is_complete(connection, "near_candidates", "frozen")
    connection.execute(
        "UPDATE m1_stage_state SET status = 'complete' WHERE stage = 'near_candidates'"
    )
    assert stage_is_complete(connection, "near_candidates", "frozen")
