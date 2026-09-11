import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _summary():
    return json.loads(
        (ROOT / "data/metadata/preparation_summary.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_corpus_accounting_reconciles_exactly():
    accounting = _summary()["corpus_accounting"]

    assert accounting["raw_core_documents"] == (
        accounting["removed_empty"]
        + accounting["removed_short"]
        + accounting["exact_duplicate_removals"]
        + accounting["exact_unique_documents"]
    )

    assert accounting["final_retained_documents"] == (
        accounting["exact_unique_documents"]
        - accounting["near_duplicate_removals"]
    )


def test_frozen_split_counts_reconcile_to_retained_corpus():
    summary = _summary()

    retained = summary["corpus_accounting"]["final_retained_documents"]
    splits = summary["split_summary"]["splits"]

    assert sum(split["documents"] for split in splits.values()) == retained
    assert splits["train"]["documents"] == 5_574_885
    assert splits["validation"]["documents"] == 309_677
    assert splits["test"]["documents"] == 309_369


def test_near_duplicate_accounting_is_self_consistent():
    near = _summary()["near_duplicates"]

    assert near["candidate_pairs_checked"] == 6_444_499
    assert near["accepted_edges"] == 86_697
    assert near["clusters"] == 13_208
    assert near["removed_documents"] == 15_253
    assert near["accepted_edges"] <= near["candidate_pairs_checked"]
