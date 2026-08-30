import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.tokenizer.corpus import build_training_corpus


def test_tokenizer_corpus_rejects_non_train_rows(tmp_path):
    path = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table(
            {
                "document_id": ["x"],
                "source": ["source"],
                "source_group": ["News"],
                "text": ["Azərbaycan dilində sınaq mətni"],
                "split": ["validation"],
            }
        ),
        path,
    )
    with pytest.raises(RuntimeError, match="non-train"):
        build_training_corpus(path, tmp_path / "train.txt", tmp_path / "manifest.parquet", 10)
