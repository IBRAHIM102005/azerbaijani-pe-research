# Corpus data card

## Source and scope

The corpus is a local subset of DOLLMA. Core components are `anl-news`, `azwiki`, `elite-blogs`, `elite-books`, `eqanun`, and `mediocore-books`. `translated-enwiki` is excluded to avoid a translated-text confound. `bhos` has status `requires_source_decision` and is not included.

The frozen inventory covers 20 original shards and 14 locally copied core shards. The authoritative inventory hash is `9c9ff30f00c3ef836f820a58be6379773ca9263a5a68957cd9fa3ce55257d6e0`. The final immutability check passed for 20 original and 14 local-core paths. The registry records 2026-08-28 as the inventory and hash revalidation date; the original acquisition date is unavailable.

## Processing policy

Text is normalized to Unicode NFC. CRLF is converted to LF, redundant horizontal whitespace is reduced, unsafe control characters are handled, and outer whitespace is trimmed. Case, punctuation, paragraph structure, and Azerbaijani letters are preserved. Documents with fewer than 50 Unicode letters are removed. Every removal appears in the accounting reports.

Exact duplicates are identified globally by SHA-256 of canonical text. Near duplicates use character 5-grams, 32-value bottom-k fingerprints, eight band files, complete pair enumeration within every observed colliding bucket, and an exact Jaccard threshold of 0.95. Accepted edges form connected components and one deterministic representative is retained per component. Component membership is transitive, so arbitrary members are not guaranteed to be pairwise above 0.95.

An independent audit found and corrected an anchor-only large-bucket candidate shortcut before model training. The repaired candidate set has 6,444,499 unique pairs. Exhaustive validation of all 59 formerly problematic bucket events captured all 321,263 eligible true pairs and missed none.

## Split policy

Duplicate clusters are assigned together with split seed 2026. Hash ranges target 90% train, 5% validation, and 5% test. The frozen counts are 5,574,885 train, 309,677 validation, and 309,369 test. The layered leakage audit status is `pass`; the independent large-bucket check found no retained true-near cross-split pair, and all 237 prerepair confirmed pairs were resolved.

## Tokenizer policy

The final tokenizer is SentencePiece BPE with 16,000 pieces and identity normalization. It was trained on a deterministic 1,000,000-document sample from train only, shared exactly by the 8K, 16K, and 32K candidates. `<unk>` is ID 0 and `<eod>` is ID 1; BOS and padding are disabled. One `<eod>` token is counted after each document. Candidate unknown rates are explicitly defined as unknown tokens divided by total SentencePiece tokens in the shared 100,000-document train audit. Internal line breaks are projected to spaces only at SentencePiece encoding time.

## Training subset policy

The train-only model corpus targets News 25%, Native Wikipedia 20%, Books 30%, Laws 15%, and Blogs 10%. Selection is deterministic, without replacement, and independent of future model seeds. When a group is exhausted, its shortage is redistributed using the frozen eligible-group weights. The selected manifest has 277,027 documents and 50,062,887 tokens. Training stops after exactly 50,000,000 token IDs, 4,185 tokens into the boundary document and before its `<eod>`.

## Reproduction

M2/M3 should consume `data/metadata/training_data_contract.json`; completed corpus and tokenizer stages should not be rerun. `python scripts/validate_frozen_corpus.py` performs a full artifact and row-reference audit when deliberate revalidation is needed. If the repository moves while the external DOLLMA clone does not, set `AZ_PE_DOLLMA_ROOT` to its new runtime location.

## Limitations

Source-level licenses and revisions are unknown where the local metadata does not state them. The dataset-level license declaration is recorded without extending it into a legal conclusion. The Books component mapping is an evidence-based inference. The minimum-length rule removes many short Books II records, quality flags do not establish language identity or OCR correctness, and connected-component endpoints are not guaranteed to meet the direct edge threshold.
