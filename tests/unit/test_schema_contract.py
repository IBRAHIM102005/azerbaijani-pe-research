import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.schema import inspect_parquet


def test_inspect_parquet_records_stable_text_schema(tmp_path):
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"text": ["Azərbaycan dili", "Bakı"]}), path)

    result = inspect_parquet(path, "azwiki", "text")

    assert result["source"] == "azwiki"
    assert result["rows"] == 2
    assert result["selected_text_column"] == "text"
    assert result["null_text_count"] == 0
    assert len(result["sha256"]) == 64
    assert len(result["schema_signature"]) == 64


def test_inspect_parquet_rejects_missing_text_column(tmp_path):
    path = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"body": ["mətn"]}), path)

    with pytest.raises(ValueError, match="no configured text column"):
        inspect_parquet(path, "source", "text")


def test_inspect_parquet_rejects_non_string_text_column(tmp_path):
    path = tmp_path / "wrong-type.parquet"
    pq.write_table(pa.table({"text": pa.array([1, 2, 3], type=pa.int64())}), path)

    with pytest.raises(TypeError, match="not a string"):
        inspect_parquet(path, "source", "text")
