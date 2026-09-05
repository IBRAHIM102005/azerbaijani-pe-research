from src.reproducibility import metadata


def test_nvidia_driver_version_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        metadata,
        "_run",
        lambda _cmd: None,
    )

    assert (
        metadata.nvidia_driver_version()
        == metadata.UNAVAILABLE
    )


def test_environment_fingerprint_uses_stable_runtime_fields(
    monkeypatch,
):
    fake_info = {
        "python_version": "3.13.0",
        "numpy_version": "2.4.0",
        "pytorch_version": "2.10.0+cu128",
        "cuda_version": "12.8",
        "cudnn_version": "91002",
        "driver_version": "570.00",
        "device_name": "NVIDIA A100-SXM4-80GB",
    }

    monkeypatch.setattr(
        metadata,
        "device_info",
        lambda: fake_info,
    )

    monkeypatch.setattr(
        metadata,
        "_bf16_supported",
        lambda: True,
    )

    assert (
        metadata.environment_fingerprint()
        == {
            **fake_info,
            "bf16_supported": True,
        }
    )


def test_collect_metadata_marks_local_model_implementation(
    tmp_path,
):
    result = metadata.collect_metadata(
        run_id="test-rope-s17",
        pe_method="rope",
        model_seed=17,
        data_seed=2026,
        resolved_config_hash=(
            "a" * 64
        ),
        repo_dir=tmp_path,
    )

    assert (
        result["model_implementation"]
        == "local_pythia_style"
    )

    assert (
        result["gpt_neox_commit"]
        == metadata.NOT_APPLICABLE
    )

    assert "driver_version" in result
    assert "cudnn_version" in result
