import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.io import source_shards, stream_source


def _write(path, texts):
    pq.write_table(pa.table({"text": texts}), path)


def test_source_shards_are_sorted_deterministically(tmp_path):
    source = tmp_path / "azwiki"
    source.mkdir()

    _write(source / "part-b.parquet", ["b"])
    _write(source / "part-a.parquet", ["a"])

    assert [path.name for path in source_shards(tmp_path, "azwiki")] == [
        "part-a.parquet",
        "part-b.parquet",
    ]


def test_stream_source_preserves_shard_and_row_order(tmp_path):
    source = tmp_path / "azwiki"
    source.mkdir()

    _write(source / "b.parquet", ["b-0"])
    _write(source / "a.parquet", ["a-0", "a-1"])

    rows = [
        (record.shard, record.row_index, record.text)
        for record in stream_source(tmp_path, "azwiki", "text")
    ]

    assert rows == [
        ("a.parquet", 0, "a-0"),
        ("a.parquet", 1, "a-1"),
        ("b.parquet", 0, "b-0"),
    ]


def test_source_shards_fail_closed_when_source_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_shards(tmp_path, "missing-source")
