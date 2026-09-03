# Reproducibility Tooling Interface Contract

This tooling owns integration & reproducibility only. It does not
reimplement Ibrahim, Yasin, Fidan, or Nihat's logic. Everything in this repo is written against the
duck-typed contract below. **Drop Ibrahim, Yasin, Fidan, and Nihat's real modules into `src/` and
wire the small adapter shims in `src/reproducibility/adapters.py` (marked
`# TODO(integration)`) to your actual import paths** — the audits, tests,
and CI will then run against real code instead of the reference fixtures
used to test this suite's own tooling.

Nothing in this repo invents experimental results, hashes, checkpoints, or
run IDs. Fixture data under `configs/frozen/resolved/*.yaml` is synthetic
and exists only to unit-test these scripts themselves (see the
`SYNTHETIC_FIXTURE` marker in each file).

## From Ibrahim (Data / Tokenizer)

**Confirmed 2026-08-31 against Ibrahim's real code and artifacts in the shared
repo** (superseding the earlier draft, which guessed wrong on three
points).

- **Primary interface: `data/metadata/training_data_contract.json`.**
  Ibrahim's own README says "Yasin/Fidan should read
  `data/metadata/training_data_contract.json`" — this tooling follows the
  same rule rather than re-deriving hashes itself. It contains, among much
  else:
  - `artifacts.manifests.{train,validation,test}.{path,sha256,bytes}`
  - `tokenizer.artifact_hashes` — `{"tokenizer.model": sha256,
    "tokenizer.vocab": sha256, "tokenizer_config.json": sha256,
    "special_tokens.json": sha256}`
  - `tokenizer.vocab_size` (16000 in the current frozen tokenizer)
  - `splits.seed` -- the train/val/test partition seed (NOT the same as
    `data_seed_from_contract()`, see below)
  - `training_subset.data_seed` -- the seed for the 50M-token training
    subset (`data/manifests/train_50m.parquet`); returned by
    `data_seed_from_contract()`
  - `training_subset.manifest_sha256` -- sha256 of `train_50m.parquet`;
    returned by `training_subset_hash_from_contract()`
  - `m1_status` — must equal `"complete"` for the contract to be trusted
  - Loaded via `src/reproducibility/adapters.py::load_training_data_contract()`,
    with typed extractors `manifest_hashes_from_contract()`,
    `tokenizer_hashes_from_contract()`, `tokenizer_vocab_size_from_contract()`,
    `data_seed_from_contract()`.
- **Manifests are Parquet, not JSONL**: `data/manifests/{train,validation,test}.parquet`,
  written against `MANIFEST_SCHEMA` in `src/data/manifests.py` (columns:
  `document_id`, `source`, `source_group`, `duplicate_cluster_id`,
  `canonical_text_hash`, `processed_file`, `processed_row`,
  `raw_record_id`, `raw_shard`, `raw_row_index`). This tooling does not
  read these files directly; it trusts the hashes already published in
  the contract above, and only re-hashes on disk (via `sha256_file`) as a
  best-effort release-verification check when the (large, often not
  checked out) Parquet files happen to be present.
- **`data/metadata/source_registry.yaml`** — not `data/source_registry.yaml`.
- **Hashing utility**: `src.data.hashing.sha256_file(path: Path,
  chunk_size: int = 8*1024*1024) -> str` (plain `hashlib.sha256(...).hexdigest()`
  over file bytes, read in chunks). Exposed via
  `src/reproducibility/adapters.py::manifest_hash_fn()`. Import path is
  `src.data.hashing` (Ibrahim's `src/` is a real package with `__init__.py`),
  not `data.hashing`.
- **`tokenizer/tokenizer_hashes.json`** — a JSON object with exactly the
  4 keys above, mirrored inside the contract's `tokenizer.artifact_hashes`.
  Both are checked by `scripts/verify_release.py`.
- **`meta.tokenizer_sha256`** in `configs/pe/*.json` and
  `src.models.data_contract.DataContract.tokenizer_sha256` are both simply
  `tokenizer.artifact_hashes["tokenizer.model"]`. `collect_metadata.py
  --from-contract` must record the same value for `tokenizer_hash`.

## From Yasin (Model / PE)

**Confirmed 2026-08-31 by reading Yasin's real code** (`src/models/`), which
had not landed when this contract was first drafted. Yasin's design is more
rigorous than an earlier placeholder here assumed, so config-drift and
parameter-fairness logic is no longer reimplemented at all — this tooling
wraps Yasin's own tested primitives instead:

- **Config format is JSON, not YAML**, one file per arm:
  `configs/pe/{learned,sinusoidal,rope,alibi,nope}.json`, plus a shared
  `configs/model_base.json` and a preregistered `configs/run_matrix.json`
  (`run_seeds: [17, 42, 1234]`, 15 total `(pe_type, init_seed)` cells).
- **`build_model(config: ModelConfig) -> PELanguageModel`** — a single
  resolved `ModelConfig` argument, not `(pe_type, config_dict)` as
  originally guessed. A config with `init_seed == TEMPLATE_SEED` (`-1`)
  cannot be built; call `.with_seed(n)` first (`n` must be one of the
  preregistered `run_seeds`).
- **Config-drift enforcement already exists**: `src.models.config.ARM_ALLOWLIST
  == frozenset({"pe_type"})` — every field of every shipped arm config is
  byte-identical except `pe_type` itself. Enforced by Yasin's own
  `tests/test_config_contract.py`, which *also* cross-checks every arm's
  `meta.manifest_sha256` / `meta.tokenizer_sha256` against Ibrahim's real
  `training_data_contract.json`. `scripts/audit_configs.py` wraps this
  rather than re-deriving an allowlist.
- **Parameter-fairness enforcement already exists**:
  `src.models.params.fairness_report(base_config, pe_types) -> dict` builds
  all five arms and checks core/embedding parameter parity *and*
  bit-identical shared-weight initialization (a per-name-seeded init
  scheme, so every arm's shared weights are provably identical, not just
  equal in count). `scripts/audit_parameters.py` wraps this rather than
  reimplementing parameter counting.
- Real, verified numbers for the frozen 6-layer/256-dim/8-head/1024-ffn/
  16k-vocab/512-context spec: baseline (non-Learned) = 12,931,072 params
  (core 4,739,072 + untied embeddings 8,192,000); Learned = 13,062,144
  (baseline + 512×256 = 131,072), exactly matching the master plan.
- `PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")` — same
  five values, now sourced from `src.models.config.PE_TYPES` rather than
  redeclared here.

## From Fidan (Training / Checkpoint)
- `save_checkpoint(path, model, optimizer, rng_state, tokens_seen, extra=None) -> None`
- `load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict`
  returning at least `{"tokens_seen": int, "rng_state": ..., "extra": ...}`
  and mutating `model`/`optimizer` in place.
- If Fidan's real function names differ, edit the single import block in
  `src/reproducibility/adapters.py::checkpoint_adapter()` — nothing else
  in the test suite should need to change, because the integration tests
  only call the adapter functions, never Fidan's internals directly.

## From Nihat (Evaluation)
- `evaluate(model, manifest_path, device="cpu") -> dict` returning at least
  `{"nll": float, "token_count": int}`. Used only by
  `make reproduce-headline` (evaluation stage), not by the unit tests here.

## Resolved configuration contract (all members)
Every experiment must produce a single resolved, canonical config
(`configs/frozen/resolved/<run_id>.yaml`) containing every field listed in
`scripts/audit_configs.py::REQUIRED_IDENTICAL_FIELDS` plus whichever fields
are in `scripts/audit_configs.py::PE_SPECIFIC_ALLOWLIST`. `audit_configs.py`
does not care which member produced which field — it only diffs the
resolved output.
