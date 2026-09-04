"""SYNTHETIC_FIXTURE: shared test fixtures for exercising this suite's own tooling.

Shaped exactly like Yasin's real config schema (src.models.config.ModelConfig
/ ARM_ALLOWLIST), confirmed 2026-08-31 against the real configs/pe/*.json
and configs/model_base.json in the shared repo. None of the numeric
values here are real experimental results.
"""
from __future__ import annotations

import copy

PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")


def base_arm_payload(pe: str, **overrides) -> dict:
    if pe not in PE_TYPES:
        raise ValueError(f"unknown PE type {pe!r}")
    payload = {
        "pe_type": pe,
        "vocab_size": 64,
        "n_layer": 2,
        "n_head": 4,
        "d_model": 32,
        "d_ff": 64,
        "max_seq_len": 16,
        "dropout": 0.0,
        "layer_norm_eps": 1e-5,
        "tie_embeddings": False,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "rope_theta": 10000.0,
        "rotary_pct": 0.25,
        "sinusoidal_theta": 10000.0,
        "alibi_max_slope_exponent": 3.0,
        "init_seed": -1,  # TEMPLATE_SEED
        "data_seed": 2026,
        "init_scheme": "pythia",
        "meta": {"note": "SYNTHETIC_FIXTURE, not a real experiment"},
    }
    payload.update(overrides)
    return payload


def all_arm_payloads(**overrides) -> dict[str, dict]:
    return {pe: base_arm_payload(pe, **overrides) for pe in PE_TYPES}


def mutate(payload: dict, key: str, value) -> dict:
    out = copy.deepcopy(payload)
    out[key] = value
    return out


def write_run_matrix(config_dir, run_seeds=(17, 42, 1234, 2027, 5003)) -> None:
    import json

    (config_dir / "run_matrix.json").write_text(
        json.dumps(
            {
                "data_seed": 2026,
                "pe_types": list(PE_TYPES),
                "run_seeds": list(run_seeds),
                "runs": [{"pe_type": pe, "init_seed": s} for s in run_seeds for pe in PE_TYPES],
            }
        )
    )
