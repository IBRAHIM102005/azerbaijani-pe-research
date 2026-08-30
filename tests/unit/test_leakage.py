from src.data.leakage import LeakageDocument, cross_split_near_pairs
from src.data.split import assign_split


def test_independent_gate_catches_omitted_cross_split_near_pair():
    common = "Azərbaycan dilində çox uzun və demək olar ki, eyni sənəd mətni. " * 20
    findings = cross_split_near_pairs(
        [
            LeakageDocument("left", "train", common + "A"),
            LeakageDocument("right", "test", common + "B"),
        ],
        threshold=0.95,
        shingle_size=5,
    )
    assert len(findings) == 1


def test_cluster_identifier_keeps_confirmed_near_pair_in_one_split():
    cluster_id = "frozen-near-cluster"
    left_split = assign_split(cluster_id, 2026, 900, 950, 1000)
    right_split = assign_split(cluster_id, 2026, 900, 950, 1000)
    assert left_split == right_split
