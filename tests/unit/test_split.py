from src.data.split import assign_split, split_bucket


def test_split_is_repeatable_and_matches_bucket_ranges():
    identifiers = [f"doc-{index}" for index in range(5000)]
    first = [assign_split(identifier, 2026) for identifier in identifiers]
    second = [assign_split(identifier, 2026) for identifier in identifiers]
    assert first == second
    for identifier, split in zip(identifiers, first):
        bucket = split_bucket(identifier, 2026)
        expected = "train" if bucket < 900 else "validation" if bucket < 950 else "test"
        assert split == expected


def test_cluster_identifier_keeps_members_in_one_split():
    cluster_id = "shared-cluster"
    assert len({assign_split(cluster_id, 2026) for _ in range(10)}) == 1
