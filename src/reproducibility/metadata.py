"""Run metadata collection.

Collects everything listed in the project plan for a single experiment
run into a stable, JSON-serializable dict.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

UNAVAILABLE = "unavailable"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception: 
        return None


def git_info(repo_dir: str | Path = ".") -> dict:
    commit = _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo_dir), "status", "--porcelain"])
    return {
        "git_commit": commit or UNAVAILABLE,
        "dirty_repository": (status != "") if status is not None else UNAVAILABLE,
    }


def timestamps() -> dict:
    now_utc = dt.datetime.now(dt.timezone.utc)
    baku = dt.timezone(dt.timedelta(hours=4))
    now_baku = now_utc.astimezone(baku)
    return {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_baku": now_baku.isoformat(),
    }


def device_info() -> dict:
    """CPU-safe device/CUDA/PyTorch info. Never raises even if torch or
    CUDA is unavailable."""
    info: dict[str, Any] = {
        "device_name": UNAVAILABLE,
        "device_uuid": UNAVAILABLE,
        "cuda_version": UNAVAILABLE,
        "pytorch_version": UNAVAILABLE,
        "peak_allocated_vram_bytes": UNAVAILABLE,
        "peak_reserved_vram_bytes": UNAVAILABLE,
    }
    try:
        import torch  # noqa: PLC0415

        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(idx)
            try:
                info["device_uuid"] = str(torch.cuda.get_device_properties(idx).uuid)
            except Exception:  # noqa: BLE001
                info["device_uuid"] = UNAVAILABLE
            info["cuda_version"] = torch.version.cuda or UNAVAILABLE
            info["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated(idx)
            info["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved(idx)
        else:
            info["device_name"] = f"cpu ({platform.processor() or platform.machine()})"
    except ModuleNotFoundError:
        pass  # torch not installed in this environment; fields stay UNAVAILABLE
    return info


def collect_metadata(
    *,
    run_id: str,
    pe_method: str,
    model_seed: int,
    data_seed: int,
    resolved_config_hash: str,
    tokenizer_hash: str | None = None,
    train_manifest_hash: str | None = None,
    validation_manifest_hash: str | None = None,
    test_manifest_hash: str | None = None,
    dataset_source_revision: str | None = None,
    precision: str | None = None,
    tokens_seen: int | None = None,
    checkpoint_hashes: dict[str, str] | None = None,
    exit_code: int | None = None,
    metrics_path: str | None = None,
    repo_dir: str | Path = ".",
) -> dict:
    """Build one experiment's metadata dict with stable key ordering.

    All hash/identifier arguments default to UNAVAILABLE (never guessed,
    never fabricated) when not supplied by the caller 
    """
    meta: dict[str, Any] = {}
    meta["run_id"] = run_id
    meta.update(timestamps())
    meta.update(git_info(repo_dir))
    meta["resolved_config_hash"] = resolved_config_hash
    meta["pe_method"] = pe_method
    meta["model_seed"] = model_seed
    meta["data_seed"] = data_seed
    meta["tokenizer_hash"] = tokenizer_hash or UNAVAILABLE
    meta["train_manifest_hash"] = train_manifest_hash or UNAVAILABLE
    meta["validation_manifest_hash"] = validation_manifest_hash or UNAVAILABLE
    meta["test_manifest_hash"] = test_manifest_hash or UNAVAILABLE
    meta["dataset_source_revision"] = dataset_source_revision or UNAVAILABLE
    meta.update(device_info())
    meta["precision"] = precision or UNAVAILABLE
    meta["tokens_seen"] = tokens_seen if tokens_seen is not None else UNAVAILABLE
    meta["checkpoint_hashes"] = checkpoint_hashes or {}
    meta["exit_code"] = exit_code if exit_code is not None else UNAVAILABLE
    meta["metrics_path"] = metrics_path or UNAVAILABLE
    return meta


def write_metadata(meta: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n")
    # Validate it round-trips as valid JSON before declaring success.
    json.loads(out_path.read_text())
