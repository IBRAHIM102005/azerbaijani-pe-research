from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.promote_corpus_repair import move_once, parquet_rows


def _write_split_directory(path: Path, rows_per_split: int) -> None:
    path.mkdir()
    table = pa.table({"document_id": [f"doc-{index}" for index in range(rows_per_split)]})
    for split in ("train", "validation", "test"):
        pq.write_table(table, path / f"{split}.parquet")


def test_invalid_staging_is_rejected_before_current_is_moved(tmp_path):
    current = tmp_path / "current"
    staged = tmp_path / "staged"
    backup = tmp_path / "backup"
    _write_split_directory(current, 1)
    _write_split_directory(staged, 1)

    with pytest.raises(RuntimeError, match="Validated staged directory"):
        move_once(current, backup, staged, expected_rows=6)

    assert current.is_dir()
    assert not backup.exists()
    assert parquet_rows(current) == 3


def test_valid_staging_is_promoted_after_validation(tmp_path):
    current = tmp_path / "current"
    staged = tmp_path / "staged"
    backup = tmp_path / "backup"
    _write_split_directory(current, 1)
    _write_split_directory(staged, 2)

    move_once(current, backup, staged, expected_rows=6)

    assert parquet_rows(current) == 6
    assert parquet_rows(backup) == 3
    assert not staged.exists()
