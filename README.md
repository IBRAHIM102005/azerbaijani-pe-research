Azerbaijani Positional Encoding Research

Research question: Which positional encoding scheme generalizes best when a small causal language model is pretrained on a limited amount of Azerbaijani text?

The study compares learned absolute embeddings, sinusoidal encoding, RoPE, ALiBi, and NoPE. The corpus, tokenizer, document order, model architecture, optimization setup, and token budget are held fixed so that positional encoding is the main experimental variable.

The data/tokenizer, model/PE, and M3 training implementations are now integrated. The frozen corpus comes from six native-Azerbaijani DOLLMA components, uses document-level duplicate-aware splits, and has one shared 16,000-piece SentencePiece BPE tokenizer. CPU end-to-end training/resume tests and five-arm model smokes pass, and development CUDA smoke/benchmark work has been exercised. The final A100 benchmark, exact A100 environment freeze, and 25-run headline training matrix are still pending, so there are no final scientific results yet.

Frozen data pipeline setup

Core sources: news, native Azerbaijani Wikipedia, blogs, laws, and two book components

Split: deterministic 90/5/5 at document/duplicate-cluster level, seed 2026

Tokenizer: SentencePiece BPE, 16,000 pieces, trained on 1,000,000 train documents only

Model-data target: one deterministic train-only sequence with an exact 50,000,000-token consumption boundary

translated-enwiki is excluded to avoid a translated-text confound. bhos remains excluded because the local upstream material does not resolve its native-source role clearly enough.

Repository structure

data/ holds local raw data, processed corpus files, frozen manifests, and data pipeline metadata.

tokenizer/ contains the final shared tokenizer and its hashes.

src/data/ and src/tokenizer/ contain the data pipeline implementation.

src/models/ contains the frozen Pythia-style causal LM and five PE variants.

src/training/ contains batching, optimizer, checkpoint/resume, runner, queue, and production guards.

src/reproducibility/ contains determinism, metadata, config, and release-integration tooling.

scripts/ contains data validation plus M3 preflight, benchmark, plan, smoke, and launcher entry points.

tests/ contains unit, integration, and repair-regression tests.

docs/notes/ contains the data pipeline analysis, data card, and independent audit.

configs/, experiments/, results/, report/, and presentation/ support later phases.

Data integrity

Raw DOLLMA parquet files stay local under data/raw/ and are not part of the repository's portable operational artifacts. The processed manifests use paths relative to the repository root. The external DOLLMA location is runtime-resolved and can be overridden with AZ_PE_DOLLMA_ROOT after moving the project.

An independent pre-experiment audit found that the first near-duplicate implementation missed non-anchor pairs in large LSH buckets. The data pipeline was repaired before model training by replacing that shortcut with complete, chunked pair enumeration for every observed bucket. Exact Jaccard verification, regenerated splits, independent leakage checks, tokenizer retraining, token recounting, and the 50M sequence rebuild were completed from the repaired state.

Validate the frozen handoff

For repository-wide development, create a clean environment with:

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproducibility.txt
python -m pip check
python -m pytest --collect-only -q
python -m pytest -q

The exact A100 training runtime is intentionally not guessed from a laptop or cloud notebook. After the A100 benchmark environment is selected, freeze it with python scripts/m3_freeze_environment.py --require-cuda --require-bf16; the resulting configs/hardware/a100_environment.json is enforced by the final M3 server preflight.

A source-only clone intentionally omits the large local manifests used by tests/integration/test_frozen_data_artifacts.py. Without the frozen artifact bundle, run the portable suite with python -m pytest -q --ignore=tests/integration/test_frozen_data_artifacts.py; the full command above applies when those local artifacts are present.

Run python scripts/validate_frozen_corpus.py from the repository root for a full read-only scan of the frozen manifests, tokenizer, token counts, processed-row references, and 50M sequence. The command refreshes only data/metadata/frozen_corpus_validation.json after all checks pass; it does not rebuild the data pipeline.

M2/M3 should read data/metadata/training_data_contract.json. Repository-internal paths resolve relative to the current project root. If the external DOLLMA clone is not at the configured relative location, set AZ_PE_DOLLMA_ROOT; a missing root produces an actionable error rather than a filesystem-wide search.

Status

Current stage: data/tokenizer, model/PE, training/resume, strict headline-plan guards, cache SHA-256 validation, shared determinism, and per-run provenance are integrated. The remaining training work is operational on the target A100: build/verify the exact 50M cache, benchmark microbatch/GAS, freeze the exact A100 runtime, regenerate the final 25-run plan, perform a real interruption/resume CUDA test, and then launch the headline matrix.

See data/README.md, docs/notes/corpus_data_card.md, and docs/notes/corpus_data_analysis.md for the measured data details.