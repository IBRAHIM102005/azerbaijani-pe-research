import pyarrow.parquet as pq


def test_synthetic_manifests_have_disjoint_ids_and_clusters(tmp_path):
    # The full synthetic pipeline test supplies the actual split writer coverage.
    from tests.integration.test_data_pipeline import _config
    from src.data.prepare import run_prepare
    import pyarrow as pa

    config = _config(tmp_path)
    raw = tmp_path / "raw" / "source-a"
    raw.mkdir(parents=True)
    pq.write_table(
        pa.table({"text": [f"Bu, sızma testi üçün uzun Azərbaycan sənədidir {index}." for index in range(300)]}),
        raw / "part.parquet",
    )
    (tmp_path / "metadata").mkdir()
    run_prepare(config, tmp_path / "interim" / "index.sqlite")
    ids = {}
    clusters = {}
    for split in ("train", "validation", "test"):
        table = pq.read_table(tmp_path / "manifests" / f"{split}.parquet")
        ids[split] = set(table.column("document_id").to_pylist())
        clusters[split] = set(table.column("duplicate_cluster_id").to_pylist())
    assert ids["train"].isdisjoint(ids["validation"] | ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert clusters["train"].isdisjoint(clusters["validation"] | clusters["test"])
    assert clusters["validation"].isdisjoint(clusters["test"])
