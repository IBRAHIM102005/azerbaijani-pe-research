import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _summary():
    return json.loads(
        (ROOT / "data/metadata/training_subset_summary.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_training_subset_has_exact_model_budget():
    summary = _summary()

    assert summary["data_seed"] == 2026
    assert summary["model_seed_affects_order"] is False

    assert summary["target_tokens"] == 50_000_000
    assert summary["selected_documents"] == 277_027
    assert summary["selected_unique_tokens"] == 50_062_887
    assert summary["overshoot_tokens"] == 62_887

    assert (
        sum(summary["actual_group_tokens"].values())
        == summary["selected_unique_tokens"]
    )


def test_exact_50m_boundary_is_inside_final_document_before_eod():
    summary = _summary()
    boundary = summary["exact_consumption_boundary"]

    assert boundary["cumulative_tokens_before_document"] == 49_995_815
    assert boundary["tokens_consumed_from_document"] == 4_185

    assert (
        boundary["cumulative_tokens_before_document"]
        + boundary["tokens_consumed_from_document"]
        == summary["target_tokens"]
    )

    assert boundary["eod_consumed"] is False
    assert boundary["unconsumed_document_tokens"] == 2_614

    assert (
        boundary["tokens_consumed_from_document"]
        + boundary["unconsumed_document_tokens"]
        == boundary["full_document_tokens_including_eod"]
    )
