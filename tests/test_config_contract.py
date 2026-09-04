"""Strict config contract across the five arms.

The study's central claim is that the arms differ in exactly one thing.  This
file is where that claim is enforced mechanically rather than asserted in
prose: every field of every shipped config is compared across all five arms,
and anything outside the allowlist that differs is a failure.

It also checks the things that make a run reproducible and traceable: the
seed policy, the packing policy, and the identity of the frozen M1 artifacts
as recorded in ``training_data_contract.json``.
"""

import json
from pathlib import Path

import pytest

from src.models.config import (
    ARM_ALLOWLIST,
    PE_OPERATIONAL_FIELDS,
    PE_TYPES,
    TEMPLATE_SEED,
    ModelConfig,
)
from src.models.data_contract import load_contract
from src.models.run_config import (
    config_sha256,
    iter_runs,
    load_run_matrix,
    resolve_run_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RUN_SEEDS = (17, 42, 1234,2027,5003)
DATA_SEED = 2026


def arm_payload(pe_type: str) -> dict:
    return json.loads(
        (CONFIG_DIR / "pe" / f"{pe_type}.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# 1. only the experimental variable may differ
# ---------------------------------------------------------------------------
def test_only_allowlisted_fields_differ_between_arms():
    payloads = {pe: arm_payload(pe) for pe in PE_TYPES}

    keys = {frozenset(p) for p in payloads.values()}
    assert len(keys) == 1, "arms do not even have the same set of config fields"

    differing = set()
    for key in next(iter(keys)):
        values = {json.dumps(p[key], sort_keys=True) for p in payloads.values()}
        if len(values) > 1:
            differing.add(key)

    assert differing == set(ARM_ALLOWLIST), (
        f"fields differing across arms: {sorted(differing)}; "
        f"allowlist is {sorted(ARM_ALLOWLIST)}"
    )


def test_pe_operational_fields_are_present_and_identical_everywhere():
    """RoPE/ALiBi/sinusoidal knobs exist in every arm with the same value.

    Only one arm reads each of them, but if they varied per arm the configs
    would no longer be a controlled set.
    """
    payloads = [arm_payload(pe) for pe in PE_TYPES]
    for field in PE_OPERATIONAL_FIELDS:
        values = {p[field] for p in payloads}
        assert len(values) == 1, f"{field} differs across arms: {values}"


@pytest.mark.parametrize("field,expected", [
    ("vocab_size", 16000),
    ("n_layer", 6),
    ("n_head", 8),
    ("d_model", 256),
    ("d_ff", 1024),
    ("max_seq_len", 512),
    ("dropout", 0.0),
    ("layer_norm_eps", 1e-5),
    ("tie_embeddings", False),
    ("init_scheme", "pythia"),
    ("rotary_pct", 0.25),
    ("rope_theta", 10000.0),
    ("sinusoidal_theta", 10000.0),
    ("alibi_max_slope_exponent", 3.0),
])
def test_frozen_architecture_field(field, expected):
    for pe_type in PE_TYPES:
        assert arm_payload(pe_type)[field] == expected, f"{pe_type}.{field}"


# ---------------------------------------------------------------------------
# 2. seed policy
# ---------------------------------------------------------------------------
def test_run_matrix_uses_the_preregistered_seeds():
    matrix = load_run_matrix()
    assert tuple(matrix["run_seeds"]) == RUN_SEEDS
    assert matrix["data_seed"] == DATA_SEED
    assert len(matrix["runs"]) == len(RUN_SEEDS) * len(PE_TYPES) == 25


def test_shipped_configs_carry_the_template_seed_not_a_production_seed():
    """Prevents configs/pe/*.json from silently becoming the run seed."""
    for pe_type in PE_TYPES:
        payload = arm_payload(pe_type)
        assert payload["init_seed"] == TEMPLATE_SEED, pe_type
        assert payload["init_seed"] not in RUN_SEEDS, pe_type
        assert payload["init_seed"] != DATA_SEED, pe_type


def test_a_template_config_cannot_be_used_to_build_a_model():
    from src.models.transformer import PELanguageModel

    template = ModelConfig.from_json(CONFIG_DIR / "pe" / "nope.json")
    assert template.is_template
    with pytest.raises(ValueError, match="template seed"):
        PELanguageModel(template)


def test_data_seed_is_constant_and_independent_of_the_model_seed():
    for pe_type in PE_TYPES:
        assert arm_payload(pe_type)["data_seed"] == DATA_SEED
    for run in iter_runs():
        assert run.config.data_seed == DATA_SEED
        assert run.init_seed in RUN_SEEDS
        assert run.config.data_seed != run.init_seed


def test_unregistered_seed_is_refused():
    with pytest.raises(ValueError, match="not a preregistered run seed"):
        resolve_run_config("nope", 2026)
    with pytest.raises(ValueError, match="not a preregistered run seed"):
        resolve_run_config("nope", 99)


# ---------------------------------------------------------------------------
# 3. packed-document position policy
# ---------------------------------------------------------------------------
def test_packing_policy_is_frozen_and_identical_in_every_arm():
    for pe_type in PE_TYPES:
        payload = arm_payload(pe_type)
        assert payload["reset_position_ids"] is False, pe_type
        assert payload["reset_attention_mask"] is False, pe_type


# ---------------------------------------------------------------------------
# 4. run identity
# ---------------------------------------------------------------------------
def test_run_ids_follow_the_master_plan_template():
    for run in iter_runs():
        assert run.run_id.startswith(f"p31az-{run.pe_type}-s{run.init_seed}-")
        assert "-t50m-c512-v16k-" in run.run_id
        assert run.run_id.endswith(run.config_sha256[:8])


def test_run_ids_are_unique_across_the_whole_matrix():
    ids = [run.run_id for run in iter_runs()]
    assert len(ids) == len(set(ids)) == 25


def test_config_hash_changes_with_the_seed_and_with_the_arm():
    a = resolve_run_config("rope", 17)
    b = resolve_run_config("rope", 42)
    c = resolve_run_config("alibi", 17)
    assert a.config_sha256 != b.config_sha256
    assert a.config_sha256 != c.config_sha256


def test_resolved_config_hash_is_stable():
    first = resolve_run_config("nope", 1234)
    second = resolve_run_config("nope", 1234)
    assert first.config_sha256 == second.config_sha256
    assert first.config_sha256 == config_sha256(second.config)


# ---------------------------------------------------------------------------
# 5. tokenizer / data identity, read from the contract (never hard-coded)
# ---------------------------------------------------------------------------
def test_configs_reference_the_contract_tokenizer_not_a_stale_path():
    contract = load_contract()
    for pe_type in PE_TYPES:
        meta = arm_payload(pe_type)["meta"]
        assert meta["tokenizer"] == contract.tokenizer_path
        assert meta["tokenizer_sha256"] == contract.tokenizer_sha256


def test_tokenizer_on_disk_matches_the_contract_hash():
    contract = load_contract()
    result = contract.verify(contract.tokenizer_path, contract.tokenizer_sha256)
    if result is None:
        pytest.skip("tokenizer/tokenizer.model not present in this checkout")
    assert result is True


def test_config_vocab_size_matches_the_frozen_tokenizer():
    contract = load_contract()
    for pe_type in PE_TYPES:
        assert arm_payload(pe_type)["vocab_size"] == contract.vocab_size


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_manifest_hashes_come_from_the_contract(split):
    contract = load_contract()
    for pe_type in PE_TYPES:
        recorded = arm_payload(pe_type)["meta"]["manifest_sha256"][split]
        assert recorded == contract.manifest_hashes[split]


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_manifest_on_disk_matches_the_contract(split):
    contract = load_contract()
    path = contract.manifest_paths[split]
    result = contract.verify(path, contract.manifest_hashes[split])
    if result is None:
        pytest.skip(f"{path} not present in this checkout")
    assert result is True


def test_fifty_million_sequence_identity_comes_from_the_contract():
    contract = load_contract()
    for pe_type in PE_TYPES:
        meta = arm_payload(pe_type)["meta"]
        assert meta["training_subset_sha256"] == contract.training_subset_sha256
        assert meta["training_subset_manifest"] == contract.training_subset_path
        assert meta["token_budget"] == contract.target_tokens == 50_000_000
        assert meta["selected_tokens"] == contract.selected_tokens


def test_fifty_million_manifest_on_disk_matches_the_contract():
    contract = load_contract()
    result = contract.verify(
        contract.training_subset_path, contract.training_subset_sha256
    )
    if result is None:
        pytest.skip("data/manifests/train_50m.parquet not present in this checkout")
    assert result is True


def test_contract_states_the_model_seed_cannot_perturb_data_order():
    contract = load_contract()
    assert contract.model_seed_affects_order is False
    assert contract.data_seed == DATA_SEED


def test_no_hash_is_hard_coded_twice_in_the_generator():
    """The generator must read hashes from the contract, not restate them."""
    source = (ROOT / "scripts" / "make_pe_configs.py").read_text(encoding="utf-8")
    contract = load_contract()
    for value in (
        contract.tokenizer_sha256,
        contract.training_subset_sha256,
        *contract.manifest_hashes.values(),
    ):
        assert value not in source, f"{value[:12]}... is hard-coded in the generator"


# ---------------------------------------------------------------------------
# 6. no dormant configuration surface
# ---------------------------------------------------------------------------
def test_pythia_is_the_only_accepted_init_scheme():
    """A dormant second scheme is a way to silently change the study."""
    from src.models.config import INIT_SCHEMES

    assert INIT_SCHEMES == ("pythia",)
    with pytest.raises(ValueError, match="unknown init_scheme"):
        ModelConfig(init_seed=17, init_scheme="normal")


def test_no_unused_init_std_field_remains():
    for pe_type in PE_TYPES:
        assert "init_std" not in arm_payload(pe_type), (
            f"{pe_type}.json still carries init_std, which nothing reads"
        )


def test_every_config_field_is_read_by_something():
    """Guards against config surface that exists but influences nothing."""
    payload = arm_payload("rope")
    known_unused = set()
    fields = set(payload) - {"meta"} - known_unused
    dataclass_fields = set(ModelConfig.__dataclass_fields__) - {"meta"}
    assert fields == dataclass_fields, (
        f"config file and ModelConfig disagree: "
        f"only in file {sorted(fields - dataclass_fields)}, "
        f"only in dataclass {sorted(dataclass_fields - fields)}"
    )
