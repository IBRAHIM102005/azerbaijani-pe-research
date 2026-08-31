"""Tests for parameter accounting and the cross-arm fairness guarantees."""

from pathlib import Path

import pytest
import torch

from src.models.config import PE_TYPES, ModelConfig
from src.models.params import (
    fairness_report,
    format_fairness_table,
    parameter_report,
    shared_init_fingerprint,
)
from src.models.transformer import PELanguageModel


def small_config(pe_type: str = "nope") -> ModelConfig:
    return ModelConfig(
        pe_type=pe_type,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        d_model=64,
        d_ff=128,
        max_seq_len=32,
    )


def test_core_parameter_count_is_identical_across_arms():
    report = fairness_report(small_config())
    cores = {row["core"] for row in report["rows"]}
    assert len(cores) == 1, cores


def test_shared_initialisation_is_bit_identical_across_arms():
    fingerprints = {
        pe: shared_init_fingerprint(PELanguageModel(small_config(pe)))
        for pe in PE_TYPES
    }
    assert len(set(fingerprints.values())) == 1, fingerprints


def test_only_the_learned_arm_owns_positional_parameters():
    report = fairness_report(small_config())
    parametric = report["arms_with_positional_parameters"]
    assert parametric == ["learned"]

    for row in report["rows"]:
        if row["pe_type"] == "learned":
            assert row["positional"] == 32 * 64
        else:
            assert row["positional"] == 0


def test_fairness_report_passes_for_a_well_formed_sweep():
    report = fairness_report(small_config())
    assert report["passed"], report["violations"]
    table = format_fairness_table(report)
    assert "PASS" in table
    for pe in PE_TYPES:
        assert pe in table


def test_fairness_report_detects_a_mismatched_arm():
    """Sanity check that the guard would actually fire."""
    base = small_config()
    rows = fairness_report(base)["rows"]
    tampered = fairness_report(base.with_pe("nope"), pe_types=("nope",))["rows"]
    deeper = ModelConfig(**{**base.to_dict(), "n_layer": 3})
    deeper_rows = fairness_report(deeper, pe_types=("nope",))["rows"]
    assert deeper_rows[0]["core"] != rows[0]["core"] == tampered[0]["core"]


def test_untied_embeddings_are_counted_separately():
    """The frozen spec uses an untied output head; both must be counted."""
    model = PELanguageModel(small_config("nope"))
    assert model.config.tie_embeddings is False
    assert model.lm_head.weight is not model.wte.weight

    report = parameter_report(model)
    assert report["input_embedding"] == 256 * 64
    assert report["output_head"] == 256 * 64
    assert report["token_embedding"] == 2 * 256 * 64
    manual = sum(
        p.numel() for p in {id(p): p for p in model.parameters()}.values()
    )
    assert report["total"] == manual


def test_tied_embeddings_are_counted_once_when_enabled():
    """The tying code path still has to be correct if the spec ever changes."""
    cfg = ModelConfig(**{**small_config("nope").to_dict(), "tie_embeddings": True})
    model = PELanguageModel(cfg)
    assert model.lm_head.weight is model.wte.weight
    report = parameter_report(model)
    assert report["token_embedding"] == 256 * 64
    assert report["output_head"] == 0


@pytest.mark.parametrize("pe_type", PE_TYPES)
def test_config_round_trips_through_json(tmp_path, pe_type):
    cfg = small_config(pe_type)
    path = tmp_path / f"{pe_type}.json"
    cfg.save_json(path)
    assert ModelConfig.from_json(path) == cfg


def test_deterministic_init_is_reproducible():
    a = PELanguageModel(small_config("rope"))
    b = PELanguageModel(small_config("rope"))
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)


# ---------------------------------------------------------------------------
# frozen-spec regression: these numbers are contractual
# ---------------------------------------------------------------------------
SPEC_BASELINE_PARAMS = 12_931_072
SPEC_LEARNED_PARAMS = 13_062_144
SPEC_CORE_PARAMS = 4_739_072


def production_config(pe_type: str = "nope") -> ModelConfig:
    """The real base config, loaded from disk, not re-declared here."""
    path = Path(__file__).resolve().parents[1] / "configs" / "model_base.json"
    return ModelConfig.from_json(path).with_pe(pe_type)


@pytest.mark.parametrize("pe_type", [p for p in PE_TYPES if p != "learned"])
def test_baseline_parameter_count_matches_the_frozen_spec(pe_type):
    report = parameter_report(PELanguageModel(production_config(pe_type)))
    assert report["total"] == SPEC_BASELINE_PARAMS
    assert report["core"] == SPEC_CORE_PARAMS


def test_learned_parameter_count_matches_the_frozen_spec():
    report = parameter_report(PELanguageModel(production_config("learned")))
    assert report["total"] == SPEC_LEARNED_PARAMS
    assert report["positional"] == SPEC_LEARNED_PARAMS - SPEC_BASELINE_PARAMS


def test_shipped_config_is_untied_and_matches_the_spec_geometry():
    cfg = production_config()
    assert cfg.tie_embeddings is False
    assert (cfg.n_layer, cfg.n_head, cfg.d_model, cfg.d_ff) == (6, 8, 256, 1024)
    assert (cfg.vocab_size, cfg.max_seq_len) == (16000, 512)
