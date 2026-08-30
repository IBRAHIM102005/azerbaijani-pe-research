from src.data.dedup import (
    PilotDocument,
    bottom_k_fingerprint,
    jaccard_similarity,
    run_near_duplicate_pilot,
)


def test_near_duplicate_similarity_and_fingerprint_are_deterministic():
    left = "Azərbaycan dilində bu uzun sınaq cümləsi eyni məzmunu daşıyır."
    right = left + " Əlavə."
    different = "Tamamilə başqa mövzuda yazılmış və kifayət qədər uzun bir cümlədir."
    assert bottom_k_fingerprint(left) == bottom_k_fingerprint(left)
    assert jaccard_similarity(left, right) > 0.80
    assert jaccard_similarity(left, different) < 0.30


def test_pilot_removes_exact_duplicates_before_near_scoring():
    text = "Bu, exact duplicate sınağı üçün kifayət qədər uzun Azərbaycan mətnidir."
    result = run_near_duplicate_pilot(
        [
            PilotDocument("a", "1", text),
            PilotDocument("b", "2", text),
            PilotDocument("a", "3", text + " Əlavə."),
        ],
        shingle_size=5,
        fingerprint_size=16,
        bands=4,
        thresholds=[0.80],
    )
    assert result["sample_exact_duplicate_documents"] == 1
    assert result["sample_cross_source_exact_groups"] == 1
    assert result["sample_documents_after_exact_dedup"] == 2
