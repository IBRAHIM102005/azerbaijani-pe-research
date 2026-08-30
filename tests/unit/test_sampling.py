import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.sampling import build_training_subset, exact_budget_boundary


def _write_counts(path):
    rows = []
    for index in range(30):
        rows.append(
            {
                "document_id": f"train-{index:03d}",
                "source": "news",
                "source_group": "News",
                "split": "train",
                "token_count": 10,
                "includes_eod": True,
            }
        )
    rows.append(
        {
            "document_id": "validation-only",
            "source": "news",
            "source_group": "News",
            "split": "validation",
            "token_count": 1000,
            "includes_eod": True,
        }
    )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_quota_sampling_is_deterministic_unique_and_train_only(tmp_path):
    counts = tmp_path / "counts.parquet"
    _write_counts(counts)
    settings = {
        "target_tokens": 100,
        "requested_group_weights": {"News": 1.0},
        "shortage_redistribution_weights": {"News": 1.0},
        "group_source_order": {"News": ["news"]},
    }
    outputs = []
    for run in range(2):
        connection = sqlite3.connect(tmp_path / f"run-{run}.sqlite")
        output = tmp_path / f"subset-{run}.parquet"
        summary = build_training_subset(connection, counts, output, settings, 2026)
        connection.close()
        table = pq.read_table(output)
        outputs.append(table.column("document_id").to_pylist())
        assert summary["selected_unique_tokens"] >= 100
        assert len(outputs[-1]) == len(set(outputs[-1]))
        assert "validation-only" not in outputs[-1]
    assert outputs[0] == outputs[1]


def test_quota_shortage_is_redistributed_without_repeating_documents(tmp_path):
    rows = [
        {
            "document_id": f"blog-{index}",
            "source": "blogs",
            "source_group": "Blogs",
            "split": "train",
            "token_count": 10,
            "includes_eod": True,
        }
        for index in range(2)
    ]
    rows.extend(
        {
            "document_id": f"news-{index}",
            "source": "news",
            "source_group": "News",
            "split": "train",
            "token_count": 10,
            "includes_eod": True,
        }
        for index in range(20)
    )
    counts = tmp_path / "counts.parquet"
    pq.write_table(pa.Table.from_pylist(rows), counts)
    connection = sqlite3.connect(tmp_path / "index.sqlite")
    output = tmp_path / "subset.parquet"
    summary = build_training_subset(
        connection,
        counts,
        output,
        {
            "target_tokens": 100,
            "requested_group_weights": {"Blogs": 0.5, "News": 0.5},
            "shortage_redistribution_weights": {"News": 1.0},
            "group_source_order": {"Blogs": ["blogs"], "News": ["news"]},
        },
        2026,
    )
    connection.close()
    selected = pq.read_table(output).to_pylist()
    assert summary["quota_shortages"]["Blogs"] == 30
    assert summary["actual_group_tokens"] == {"Blogs": 20, "News": 80}
    assert len({row["document_id"] for row in selected}) == len(selected)


def test_exact_budget_boundary_records_inside_document_cutoff():
    boundary = exact_budget_boundary(
        [
            {"document_id": "a", "token_count": 7},
            {"document_id": "b", "token_count": 5},
        ],
        10,
    )
    assert boundary["sampling_order_zero_based"] == 1
    assert boundary["cumulative_tokens_before_document"] == 7
    assert boundary["tokens_consumed_from_document"] == 3
    assert boundary["unconsumed_document_tokens"] == 2
    assert not boundary["eod_consumed"]
