# Azerbaijani Positional Encoding Research

**Research question:** Which positional encoding scheme generalizes best when a small causal language model is pretrained on a limited amount of Azerbaijani text?

The study compares learned absolute embeddings, sinusoidal encoding, RoPE, ALiBi, and NoPE. The corpus, tokenizer, document order, model architecture, optimization setup, and token budget are held fixed so that positional encoding is the main experimental variable.

The data and tokenizer foundation is complete. The frozen corpus comes from six native-Azerbaijani DOLLMA components, uses document-level duplicate-aware splits, and has one shared 16,000-piece SentencePiece BPE tokenizer. No model training or positional-encoding experiment has started, so there are no experimental results yet.

## Frozen data pipeline setup

- Core sources: news, native Azerbaijani Wikipedia, blogs, laws, and two book components
- Split: deterministic 90/5/5 at document/duplicate-cluster level, seed 2026
- Tokenizer: SentencePiece BPE, 16,000 pieces, trained on 1,000,000 train documents only
- Model-data target: one deterministic train-only sequence with an exact 50,000,000-token consumption boundary

`translated-enwiki` is excluded to avoid a translated-text confound. `bhos` remains excluded because the local upstream material does not resolve its native-source role clearly enough.

## Repository structure

- `data/` holds local raw data, processed corpus files, frozen manifests, and data pipeline metadata.
- `tokenizer/` contains the final shared tokenizer and its hashes.
- `src/data/` and `src/tokenizer/` contain the data pipeline implementation.
- `scripts/` contains inspection, repair, validation, and artifact-generation entry points.
- `tests/` contains unit, integration, and repair-regression tests.
- `docs/notes/` contains the data pipeline analysis, data card, and independent audit.
- `configs/`, `experiments/`, `results/`, `report/`, and `presentation/` support later phases.

## Data integrity

Raw DOLLMA parquet files stay local under `data/raw/` and are not part of the repository's portable operational artifacts. The processed manifests use paths relative to the repository root. The external DOLLMA location is runtime-resolved and can be overridden with `AZ_PE_DOLLMA_ROOT` after moving the project.

An independent pre-experiment audit found that the first near-duplicate implementation missed non-anchor pairs in large LSH buckets. The data pipeline was repaired before model training by replacing that shortcut with complete, chunked pair enumeration for every observed bucket. Exact Jaccard verification, regenerated splits, independent leakage checks, tokenizer retraining, token recounting, and the 50M sequence rebuild were completed from the repaired state.

## Validate the frozen handoff

The data-pipeline dependency set is tested on CPython 3.13 and 3.14. Set up a clean environment with:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-data-pipeline.txt
python -m pip check
python -m pytest --collect-only -q
python -m pytest -q
```

A source-only clone intentionally omits the large local manifests used by `tests/integration/test_frozen_data_artifacts.py`. Without the frozen artifact bundle, run the portable suite with `python -m pytest -q --ignore=tests/integration/test_frozen_data_artifacts.py`; the full command above applies when those local artifacts are present.

Run `python scripts/validate_frozen_corpus.py` from the repository root for a full read-only scan of the frozen manifests, tokenizer, token counts, processed-row references, and 50M sequence. The command refreshes only `data/metadata/frozen_corpus_validation.json` after all checks pass; it does not rebuild the data pipeline.

M2/M3 should read `data/metadata/training_data_contract.json`. Repository-internal paths resolve relative to the current project root. If the external DOLLMA clone is not at the configured relative location, set `AZ_PE_DOLLMA_ROOT`; a missing root produces an actionable error rather than a filesystem-wide search.

## Status

Current stage: the data and tokenizer artifacts are frozen and validated for M2/M3 handoff. Model implementation and training remain out of scope for this stage.

See [`data/README.md`](data/README.md), [`docs/notes/corpus_data_card.md`](docs/notes/corpus_data_card.md), and [`docs/notes/corpus_data_analysis.md`](docs/notes/corpus_data_analysis.md) for the measured data details.
