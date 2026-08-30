# Data

The core corpus is built from local DOLLMA parquet shards. Raw material is immutable and stays under `data/raw/DOLLMA/`; it is not a portable operational artifact.

## Source selection

The frozen native-Azerbaijani corpus contains:

- `anl-news` — News
- `azwiki` — Native Wikipedia
- `elite-blogs` — Blogs
- `elite-books` — Books
- `eqanun` — Laws
- `mediocore-books` — Books

`translated-enwiki` is excluded because translated text would add a source and language confound. `bhos` remains `requires_source_decision`; the local DOLLMA documentation does not establish its source role clearly enough for inclusion.

The local upstream README declares DOLLMA under CC BY-NC-SA 4.0. Per-component licenses and revisions are unavailable in the local upstream metadata and are recorded as unknown rather than inferred. The source registry records the access date, shard identifiers, and SHA-256 snapshot hashes.

## Frozen data pipeline artifacts

- `raw/` contains unchanged local source material.
- `interim/` contains resumable indexes and preserved pre-repair evidence.
- `processed/corpus/` contains the retained train, validation, and test text in Parquet.
- `manifests/` contains portable split and 50M-sequence references.
- `metadata/` contains inventories, hashes, accounting, audits, token counts, and the data pipeline handoff.

Cleaning is conservative: NFC normalization, CRLF handling, redundant horizontal-space cleanup, unsafe-control handling, and outer trimming. Azerbaijani spelling, case, punctuation, and paragraph boundaries are preserved. Documents below the frozen 50-Unicode-letter threshold are removed with source-level accounting.

Exact duplicates are grouped globally by canonical-text SHA-256. Near duplicates use character 5-grams, complete pair emission inside every observed LSH bucket, and exact Jaccard acceptance at 0.95. Accepted edges form connected components; transitive members are not necessarily pairwise at or above 0.95. One deterministic representative is retained per component.

The final manifests use a deterministic cluster-aware 90/5/5 split with seed 2026. The shared SentencePiece BPE tokenizer was fitted on a deterministic 1,000,000-document subset of train only. The source-aware model sequence is also train-only and fixes document order plus the exact 50,000,000-token stopping boundary.

An independent audit identified and corrected the original large-bucket candidate-generation shortcut before model training. The repaired candidate set, split, tokenizer, token counts, and 50M sequence supersede the preserved pre-repair artifacts.
