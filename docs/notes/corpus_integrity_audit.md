# M1 Final Independent Audit

Audit time: 2026-08-27 17:19 UTC

Scope: frozen M1 data, tokenizer, provenance, source code, tests, manifests, token counts, 50M selection, handoff, and Definition-of-Done evidence. Raw preprocessing, tokenizer training, and 50M selection were not rerun. No frozen research artifact was changed.

## Executive verdict

Overall M1 quality: **78.0 / 100**

- Scientific readiness: 72%
- Engineering readiness: 70%
- Reproducibility readiness: 68%
- M2/M3 handoff readiness: 45%
- Code quality: 78%
- Data integrity confidence: 74%
- Tokenizer confidence: 97%
- Current 50M corpus confidence: 92%
- Defense/report readiness: 68%

Final verdict: **NOT READY**

Definition-of-Done reassessment: **NOT COMPLETE**

There is no P0 blocker, but there is one P1 scientific-integrity defect. Candidate generation uses a star expansion in large LSH band buckets. Exhaustive checking of eight stratified large bucket-events measured only 1.83% direct recall for true pairs that satisfy the frozen near-duplicate rule. More importantly, the same sample exposed 237 retained representative pairs in different splits that still satisfy that rule. The existing leakage report proves that accepted edges and known clusters do not cross splits; it does not prove that omitted true near pairs do not cross splits.

The raw-normalization and exact-dedup work remains usable. The correction can start from verified fingerprints/band files, but near-candidate verification, clustering, splitting, and every dependent frozen artifact must be rebuilt before M2/M3.

One scoring specification issue was handled explicitly: the supplied category weights add to 105, not 100. No weight was altered. The weighted numerator was divided by 105, producing 77.9714, rounded to 78.0.

## What was independently verified

- The M1-relevant master-plan requirements and frozen configuration were read against actual code and artifacts.
- The SQLite state was queried read-only. Its document, removal, candidate, edge, cluster, and representative counts agree with persisted reports.
- All eight near-band files were scanned. Large-bucket behavior was measured from the frozen data.
- Train, validation, and test Parquet manifests were scanned directly. Their hashes, ordering, row references, IDs, canonical hashes, and known cluster assignments were checked.
- Tokenizer candidate metrics were independently recomputed on the same 100,000 train documents for 8K, 16K, and 32K.
- The final 16K model was loaded and its model type, vocabulary, normalization, special tokens, Azerbaijani handling, and SHA-256 values were checked.
- The one-million-document tokenizer-training sample was scanned against the processed train corpus.
- Full token-count aggregates were checked and 90 stratified real documents were retokenized independently.
- The 50M selection algorithm was replayed without writing a replacement manifest. IDs, order, quotas, shortage redistribution, totals, and the exact 50M boundary matched.
- Thirty-four raw paths and 31 critical artifact files were rehashed. No size or hash mismatch was found.
- The current test suite was run independently: 18 passed, 0 failed, with two SWIG deprecation warnings in 7.03 seconds.

## Requirement compliance matrix

| Requirement | Specification / implementation | Proof | Independent result | Status |
| --- | --- | --- | --- | --- |
| Native Azerbaijani core; translated data excluded | Master plan, `configs/frozen/m1.yaml`, source registry | Inventory and processed-source scan | Six native components retained; `translated-enwiki` excluded; `bhos` unresolved and excluded | PASS |
| Conservative normalization | Master plan, `src/data/normalize.py` | Tests and preparation state | NFC and formatting cleanup preserve case and Azerbaijani letters | PASS |
| Exact dedup and traceability | Master plan, exact-dedup code and SQLite | Exact reports and SQL | 144,480 removals; accounting and deterministic representatives agree | PASS |
| Near dedup and leakage protection | Master plan, `src/data/near.py` | Bands, candidates, edges, clusters | Large-bucket star expansion misses true pairs and leaves sampled cross-split near pairs | FAIL |
| Deterministic 90/5/5 split | Master plan, `src/data/split.py` | Three frozen manifests | 90.005476% / 4.999764% / 4.994759%; known groups intact | PASS |
| Hard leakage gate | Master plan, audit scripts | Leakage JSON and direct manifest scan | Standard ID/hash/accepted-edge checks pass; broader true-near check fails | FAIL |
| Same train-only tokenizer corpus | Master plan, tokenizer corpus/train code | Sample manifest and candidate metadata | One deterministic input and matching hashes for 8K/16K/32K | PASS |
| Frozen final 16K BPE | Master plan, tokenizer train code | Root `tokenizer/` artifacts | Valid 16,000-piece SentencePiece BPE and matching hashes | PASS |
| Real tokenizer-based counts | Master plan, tokenizer count code | Per-document index and aggregate report | 606,218,773 total tokens, including one EOD per document | PASS |
| Deterministic train-only 50M mixture | Master plan, `src/data/sampling.py` | `train_50m.parquet` and summary | Replay matched all 276,565 documents and 50,069,724 full-document tokens | PASS |
| Provenance and raw immutability | Master plan, inventory/immutability tools | Inventory, registry, raw hashes | Zero raw mismatches; access date is missing | WARNING |
| Frozen M2/M3 handoff | Master plan, handoff/finalization scripts | Handoff and DoD JSON | Numeric values agree, but completion/leakage state is stale and paths are non-portable | FAIL |

## Data pipeline audit

The frozen accounting is internally exact:

```text
8,227,654 raw
- 1,873,990 short
-   144,480 exact duplicates
= 6,209,184 exact-unique

6,209,184 exact-unique
-    15,132 near-duplicate removals
= 6,194,052 retained

5,574,986 train
+ 309,688 validation
+ 309,378 test
= 6,194,052 retained
```

The removal categories are represented as sequential and mutually reconcilable stages. Empty-document removal is zero. The 50-Unicode-letter threshold is configured and deterministic. It removes 1,873,780 `mediocore-books` records, so its domain effect is substantial even though many of those records are short fragments. This is a disclosed design consequence, not evidence that the filter is intrinsically wrong. The final paper should show source counts before and after this rule.

Normalization is conservative: NFC, line-ending cleanup, redundant horizontal-space handling, safe control removal, and outer trimming. It does not lowercase, transliterate Azerbaijani characters, stem, lemmatize, strip suffixes, or remove stop words. Per-transformation counts from the resumed preparation run remain `unknown_resumed`; removal accounting is unaffected, but transformation-frequency reporting is incomplete.

Provenance facts are separated as follows:

- **Known:** DOLLMA is the dataset; the local README declares CC BY-NC-SA 4.0; 20 original and 14 core local shards are inventoried; source/shard IDs, byte sizes, schemas, rows, and raw SHA-256 values are present; `translated-enwiki` is excluded; `bhos` is marked `requires_source_decision`.
- **Unknown:** dataset revision, per-source revisions, and per-source license terms where the local upstream metadata is silent.
- **Inferred and disclosed:** the Books I / Books II association from published size descriptions. Component identity remains preserved.
- **Missing:** an explicit frozen access date.

Independent rehashing covered 34 original/local raw paths, 8,994,376,464 bytes, with zero hash or size discrepancies. The canonical inventory hash is `9c9ff30f00c3ef836f820a58be6379773ca9263a5a68957cd9fa3ce55257d6e0`.

## Exact dedup audit

Exact grouping uses canonical normalized text and SHA-256. Persistent document IDs are source-aware, and Python's unstable `hash()` is not used. A canonical-hash uniqueness constraint detects duplicates across the complete core ingestion stream. The retained record follows deterministic ingestion order; no cross-source exact group exists in this corpus, so source preference does not affect the observed result.

SQLite and reports agree on 107,267 exact groups and 144,480 removed records. Cross-source exact groups are zero. Provenance for representatives and removed members remains available. No evidence of unique-record loss was found in the accounting or sample checks.

## Near-dedup audit

The implementation creates 32-value bottom-k fingerprints from character 5-grams, divides them into eight four-value bands, and hashes band keys with SHA-256. Candidate pairs are verified with the configured exact Jaccard and length-ratio rules. The accepted threshold is 0.95. Candidate de-duplication uses a SQLite primary key.

Persisted values are real and consistent:

- 4,191,262 unique candidate pairs
- 36,333 accepted edges
- 28,322 clustered documents
- 13,190 connected components
- 15,132 removals

Every accepted edge has Jaccard at least 0.95. The observed accepted-edge similarity range is 0.95 to 1.0.

### Large-bucket candidate recall

Bucket expansion has three branches:

- Up to 200 members: enumerate all pairs.
- 201 through 10,000 members: emit only pairs from the minimum-rowid anchor to every other member.
- Above 10,000: record the skip; the preparation runner is designed to fail if any skip exists.

Anchor choice is deterministic because bucket members and ingestion rowids are deterministic. It is not content-derived. An omitted pair may still be recovered if it collides in another band.

The frozen eight-band scan found:

| Measure | Value |
| --- | ---: |
| Colliding band-bucket events | 728,772 |
| Full-expansion bucket-events | 728,713 |
| Star bucket-events | 59 |
| Skipped bucket-events | 0 |
| Unique documents in star buckets | 10,751 |
| Maximum bucket size | 940 |
| Full-pair opportunities across star bucket-events | 3,503,354 |
| Star-emitted pair attempts | 18,678 |

The opportunity count is per band-bucket event and can count the same document pair in more than one band.

Eight star bucket-events were sampled across the observed size range: 201, 247, 291, 352, 407, 623, 799, and 940. All 1,190,954 unique pairs among 3,857 documents were evaluated with the real frozen criterion.

| Sample result | Value |
| --- | ---: |
| True Jaccard ≥ 0.95 pairs | 28,850 |
| Eligible after length-ratio gate | 28,849 |
| Direct accepted candidates | 527 |
| Missed eligible true pairs | 28,322 |
| Measured direct candidate recall | 1.8268% |

This is **measured sample recall**, not a corpus-wide estimate. One coherent 352-document laws bucket contributes most true pairs, so the sample cannot support a global percentage. It does show that the approximation is unsafe for at least one real domain pattern.

Existing connected components protected 23,504 of the 28,849 eligible sample pairs; 5,345 fell into different frozen clusters. Among the affected bucket-events, 237 unique retained representative pairs still satisfy the near criterion while belonging to different splits. Their measured similarities range from 0.950026 to 0.976311. This is not merely a theoretical false-negative risk.

Severity: **P1 — CRITICAL**. The defect must be fixed before M2/M3. The observed maximum bucket size is 940, so complete expansion for the actual large buckets would add at most 3.5 million band-bucket pair opportunities before cross-band de-duplication. That is material but tractable relative to the current 4.19 million candidate set; correctness should take priority over the shortcut.

### Connected-component transitivity

Near groups are connected components, and each keeps the minimum SHA-256 document ID. Therefore `28,322 - 13,190 = 15,132` is the correct removal equation for this implementation.

- 12,494 clusters have size two.
- 696 clusters have size greater than two.
- Maximum cluster size is 359.
- In the 20 largest clusters, 17 contain a sampled endpoint pair below 0.95.
- In the same set, 19 contain a representative/member pair below 0.95.
- Minimum observed endpoint similarity is 0.819923.
- Minimum observed representative/member similarity is 0.858268.

This is expected from graph transitivity: A–B and B–C can pass even when A–C does not. Connected closure is useful for leakage containment when candidate recall is adequate, but it is more aggressive than complete-link clustering. The global effect is limited—15,132 documents, about 0.244% of the exact-unique corpus—but the semantics need to be stated in methods and limitations.

## Split and leakage audit

The direct manifest scan reproduced these SHA-256 values:

| Split | Documents | Percent | SHA-256 |
| --- | ---: | ---: | --- |
| Train | 5,574,986 | 90.005476% | `7a597d072fd2aeb3b4df0741390fdc8fc296d57c7b783f86760044747fd2296c` |
| Validation | 309,688 | 4.999764% | `86ecafec374c9b560b4bd2a40e2fe1276ed8737b58f4d19fffef29987ea58741` |
| Test | 309,378 | 4.994759% | `e3d827093061396745ccb0d3edad3932d955b325bd8da268b603d0baa06e1521` |

Document IDs, canonical hashes, and recorded duplicate-cluster IDs are unique where required and have zero cross-split intersections. All processed row references resolve. All 36,333 accepted near edges remain inside their recorded clusters and splits. Per-source proportions remain close to 90/5/5; `elite-books` is visibly noisier because it has only 97 retained documents.

Those standard checks pass, but they only cover detected groups. The large-bucket audit independently found retained true-near pairs across splits. Accordingly, the existing `leakage_gate_passed: true` conclusion is too narrow and the hard research gate fails.

The tokenizer sample is train-only. The 50M subset is train-only and has no repeated selected ID. There was no evidence that validation or test text affected tokenizer selection, filtering, or source quotas.

## Tokenizer audit

All three SentencePiece candidates use the same deliberate one-million-document subset: the first 1,000,000 rows of the document-ID-sorted processed train corpus. The scan found no non-train ID, ordering error, or mismatch with the processed train rows. The sample is not source-balanced—it is dominated by `mediocore-books`—but it is deterministic, fixed before candidate comparison, shared by all candidates, and documented as a tokenizer-training subset.

- Training-sample manifest SHA-256: `b33353f1b13ab9fd95523299ac70155dacbf095c92ae6c1c2cbf0bd68855bca4`
- SentencePiece input SHA-256: `137f4c2e4d43802f18ac81e615ce57119ad07e8487e763bd85deaf9b13284605`

Independent candidate recomputation on the same 100,000 train documents matched the persisted metrics exactly:

| Vocabulary | Tokens | Fertility | Characters/token | Unknown pieces | Unknown rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 11,131,949 | 1.921140 | 4.086323 | 1 | 8.98315e-8 |
| 16K | 9,701,500 | 1.674275 | 4.688836 | 1 | 1.03077e-7 |
| 32K | 8,741,218 | 1.508550 | 5.203936 | 1 | 1.14401e-7 |

The final tokenizer is a 16,000-piece SentencePiece BPE trained with identity normalization, character coverage 1.0, byte fallback disabled, input shuffling disabled, and one trainer thread. SentencePiece version is 0.2.1. Special-token behavior is:

- `<unk>`: ID 0
- `<eod>` / EOS: ID 1
- BOS: disabled, ID -1
- PAD: disabled, ID -1

The model and vocabulary hashes verify:

- `tokenizer.model`: `e28256d22ee0ce47a3fbdf223bf40297094eede567139f44ef7f65a4b967b7dc`
- `tokenizer.vocab`: `aed49d8684f19c2d5a44ec5bdfc28dc1b08cec5839d44071414ac16ff644bf1b`

`ə Ə ı I İ ö Ö ü Ü ş Ş ç Ç ğ Ğ` round-trip without `<unk>`. Representative suffix-rich words, including `şəhərlərimizdəkilərdən`, `öyrənəcəklərimizlə`, and `müstəqilliyimizdəkilərin`, also round-trip. These checks support coverage, not a claim that BPE boundaries are morphemes.

The unknown-rate denominator needs clearer labeling. Candidate rates are unknown pieces divided by encoded pieces on the 100,000-document audit sample, excluding EOD. Full-corpus storage contains 606,218,773 tokens, including 6,194,052 EOD tokens. The comparable full-text denominator is therefore 600,024,721 pieces: 2,873 / 600,024,721 = 4.788136e-6. Dividing by the stored EOD-inclusive total gives 4.739213e-6. The underlying values are not wrong; the reports should state which denominator is used before putting these rates side by side.

## Token-count audit

The frozen final tokenizer produced:

| Split | Tokens | Unknown pieces |
| --- | ---: | ---: |
| Train | 544,915,436 | 2,712 |
| Validation | 30,383,019 | 46 |
| Test | 30,920,318 | 115 |
| Total | 606,218,773 | 2,873 |

Each document count includes one appended EOD. Split totals add exactly, source totals add to the retained total, and every one of the 6,194,052 retained documents has a token-count row. Independent retokenization of 90 source/split-stratified documents produced no mismatch.

Source totals are:

| Source | Tokens |
| --- | ---: |
| `anl-news` | 237,559,421 |
| `azwiki` | 74,609,017 |
| `elite-blogs` | 1,528,402 |
| `elite-books` | 7,950,104 |
| `eqanun` | 65,141,660 |
| `mediocore-books` | 219,430,169 |

## 50M selection replay

The independent replay used the frozen token-count index and data seed 2026. It did not write a replacement subset. All 276,565 document IDs, source/group labels, token counts, phases, ranks, and ordering entries match `data/manifests/train_50m.parquet`. IDs are unique and train-only.

| Group | Requested tokens | Selected tokens | Actual share |
| --- | ---: | ---: | ---: |
| News | 12,500,000 | 13,425,136 | 26.812882% |
| Native Wikipedia | 10,000,000 | 10,737,941 | 21.445976% |
| Books | 15,000,000 | 16,105,540 | 32.166225% |
| Laws | 7,500,000 | 8,394,683 | 16.765986% |
| Blogs | 5,000,000 | 1,406,424 | 2.808931% |

Blogs exhausts all eligible unique train material, leaving a 3,593,576-token shortage. The shortage is redistributed deterministically without replacement. Total selected full-document tokens are 50,069,724; the 69,724-token overshoot is reported honestly. Manifest SHA-256 is `248b4462beb043e592993e69e858f02d102d955dbab638f5509c112927e3de8d`.

The exact 50,000,000th consumed-token boundary is reproducible:

- Zero-based sampling order: 276,157 (sequence position 276,158)
- Document ID: `6aba7eed1cd6eb1ba6b677b9bf250d260af0418ca1c93cfa2db5b1e48afcce0b9`
- Document token count including EOD: 1,954
- Cumulative tokens before it: 49,998,512
- Tokens consumed from it: 1,488
- Tokens left in it: 466, consisting of 465 text tokens plus EOD

M3 must stream documents in manifest order, apply the frozen text projection and tokenizer, append exactly one EOD after each completed document, and stop after emitting the first 50,000,000 token IDs. The cutoff occurs inside text, before that document's EOD. This exact boundary should be placed in the corrected handoff rather than left merely derivable.

The current 50M artifact is algorithmically strong, but it depends on the split that failed the broader near-leakage gate. It must be refrozen after the near-dedup correction.

## Reproducibility audit

Strong points include frozen seeds, package versions, config hashes, raw hashes, manifest hashes, tokenizer hashes, deterministic selection rules, and streaming entry points. Critical artifacts rehashed with zero mismatches.

Weak points are relocation, incomplete intermediate binding, and resume validation. Band files are checked by expected count and size but are not content-hashed into the handoff. Some output writers publish directly rather than through a validated temporary replacement. The preparation skip gate can trust summary/database existence without a complete integrity replay. An explicit `--rebuild` path is user-controlled, but recursive deletion should still prove its resolved targets remain inside configured M1 output roots.

The environment is recorded as Python 3.13.14, SentencePiece 0.2.1, PyArrow 25.0.1, NumPy 2.4.0, datasketch 2.0.0, PyYAML 6.0.3, matplotlib 3.10.8, and pytest 8.4.2. Hardware-specific results are not embedded in deterministic choices.

## Portability audit

**Safe to move now: CONDITIONAL — not as-is.**

At least 183 absolute path occurrences exist in M1-relevant text metadata. More importantly, every `processed_file` value in train, validation, test, and `train_50m.parquet` points to the current OneDrive repository. The handoff and tokenizer metadata also record absolute locations. Scripts generally resolve the repository root at runtime, but `original_dollma: ../DOLLMA` assumes a sibling layout and would not resolve the current Desktop DOLLMA clone after moving only the repository to `C:\Research`.

Path classes:

- **Portable:** config values already relative to the repository, hashes, IDs, row numbers, and source labels.
- **Runtime-resolved:** most script roots derived from the script location.
- **Absolute but harmless:** historical paths used only as provenance, provided consumers never open them.
- **Absolute and breaking:** manifest `processed_file` columns and any handoff/config field consumed as a live path.

Migration procedure:

1. Complete the near-dedup correction and downstream refreeze first.
2. Store processed references relative to the repository, or add one documented relocation-aware resolver used by all consumers.
3. Set DOLLMA through an explicit path or preserve a verified sibling layout.
4. Regenerate affected hashes, handoff, reports, and DoD once.
5. Run hash, leakage, tokenizer-load, subset-stream, and pytest gates before M2/M3.

## Code quality / humanization

- Code correctness: 72%
- Human-written quality: 86%
- Readability: 84%
- Maintainability: 72%
- Performance quality: 80%

The core modules are generally direct and human-readable. Comments are sparse and usually explain decisions. No tutorial-style or obvious AI comment spam was found. Naming and type boundaries are mostly clear.

The cleanest files are `src/data/hashing.py`, `src/data/normalize.py`, `src/data/split.py`, `src/tokenizer/corpus.py`, and `src/tokenizer/counts.py`.

The most problematic correctness file is `src/data/near.py` because its performance shortcut changes candidate recall. `scripts/generate_m1_report.py`, `scripts/generate_m1_handoff.py`, and `scripts/finalize_m1_audit.py` contain long main functions and repeated artifact assembly. That is a maintainability issue, not a reason for an aesthetics-only rewrite. Some scripts named as audits also write finalized metadata; their CLI documentation should distinguish read-only validation from mutation.

Static parsing covered 39 Python files with no syntax error. No `.env` file or obvious token/secret pattern was found in the reviewed scope.

## Performance audit

Current performance quality: 80%.

Already optimal enough:

- Bounded Parquet scanning and batched SQLite operations
- Memory-mapped band files
- Parallel token counting with deterministic aggregation
- Single-thread SentencePiece fitting for reproducibility
- Hash-based, stable sampling without full-corpus in-memory materialization

Worth fixing, high value:

- Replace the large-bucket star shortcut with recall-preserving candidate expansion. The frozen data's 59 star buckets create about 3.5 million full-pair opportunities, which is not an O(N²) corpus-wide operation and appears tractable.

Potential optimization, low value:

- Reduce repeated metadata/report scans and share pure artifact builders.

Not worth touching before experiments:

- GPU conversion, aesthetic refactors, or micro-optimizing small JSON writers.

No new runtime benchmark was performed; these judgments come from code paths and persisted workload sizes.

## Test adequacy

Test pass status: 100% (18/18).

Test adequacy: 55%.

The suite covers normalization, hashing, basic near similarity/pilot behavior, deterministic split, quota mechanics, a tiny end-to-end flow, train-only rejection, frozen tokenizer/sample checks, and the frozen 50M artifact. The assertions are meaningful.

It does not cover:

- Buckets above 200 members and star candidate recall
- True-near pairs omitted from accepted clusters
- Connected-component transitivity semantics
- Exhaustive cross-split near checks on a frozen sample
- Exact 50M inside-document/EOD boundary
- Resume validation and partial-state rejection
- Raw-immutability tool behavior
- Repository relocation and path resolution
- Handoff schema/status invalidation after a failed scientific gate

The most important defect therefore passed all tests. Add regression tests before the focused rerun; do not weaken existing tests.

## Artifact consistency

The numerical values and hashes in preparation, duplicate, manifest, tokenizer, token-count, 50M, handoff, and DoD artifacts agree with one another. Thirty-one critical artifact files rehashed without mismatch.

Their semantics are no longer consistent with the independent evidence. `m1_handoff.json` says `m1_status: complete`, the DoD says complete, and the leakage artifacts say pass. Those claims are based on accepted edges and recorded clusters, while this audit found true near pairs omitted by candidate generation. The audit did not modify those frozen files merely to make them agree.

The exact 50M stop position, unknown-rate denominators, and connected-component all-pairs limitation are not explicit enough in the current handoff/report set. Near-band content hashes and some large intermediate evidence are also absent from the handoff.

## Documentation consistency

`README.md` and `data/README.md` still describe the repository as being before preprocessing and tokenizer work. That is stale. The M1 analysis note and data card contain the measured final counts, but they repeat the narrow leakage-pass claim and do not disclose the star-bucket approximation or connected-component topology.

Documentation should be updated only after corrected artifacts are frozen. It must not preserve the current `M1 COMPLETE` wording.

## Report / defense readiness

Four existing figures and the measured tables cover source documents, source/group tokens, tokenizer fertility, and the requested/actual 50M mixture. They are useful for an IEEE-style methods and data section.

Before defense or paper submission, add or update only evidence-bearing material:

- Source counts before and after the 50-letter filter
- Exact and corrected near-dedup accounting
- Corrected leakage result and candidate-recall method
- Split distribution by source
- Tokenizer comparison with explicit denominators
- Exact 50M consumption policy and Blog shortage
- Known/unknown/inferred provenance table

Do not claim that BPE discovers morphemes or that every pair in a connected component has similarity at least 0.95.

## Scientific confidence check

| Question | Answer | Evidence |
| --- | --- | --- |
| Is M1 scientifically valid for the intended PE comparison? | PARTIAL | Controlled inputs are sound; near leakage must be corrected. |
| Could data leakage bias future PE results? | YES | 237 retained cross-split representative near pairs were measured in the sample. |
| Could tokenizer construction bias one PE variant over another? | NO | Every PE condition is assigned the same frozen tokenizer. |
| Is the same data/order guaranteed for all PE variants? | YES | Selection is seed-2026/hash-frozen and independent of model seed, after required refreeze. |
| Are low-data conditions reproducible? | PARTIAL | The 50M sequence replays, but current paths and split need correction. |
| Is translated data correctly excluded? | YES | `translated-enwiki` is absent from core processing and tokenizer input. |
| Is dedup aggressive enough? | NO | Large-bucket recall is materially incomplete. |
| Is dedup too aggressive? | PARTIAL | Connected closure removes some sub-threshold endpoint pairs; global near-removal fraction is small. |
| Could the short filter distort domains? | YES | It removes 1,873,780 `mediocore-books` rows. |
| Is 16K defensible? | YES | It is preregistered, coverage-safe, and independently audited. |
| Is the current 50M corpus defensible? | PARTIAL | Its selection is exact, but it inherits the split requiring refreeze. |
| Can M2/M3 start without rerunning M1? | NO | A focused near-dedup and downstream rerun is required. |
| Can the repository move out of OneDrive now? | NO | Live manifest paths are absolute. |
| Are there absolute-path risks? | YES | Data-consuming fields point to the current OneDrive path. |
| Could this invalidate a later paper? | YES | Proceeding with the known near-leakage issue would weaken evaluation integrity. |

## Weighted scorecard

| Area | Weight | Score | Status | Confidence |
| --- | ---: | ---: | --- | --- |
| Research design compliance | 8 | 85.0% | WARNING | HIGH |
| Source/provenance | 5 | 82.0% | WARNING | HIGH |
| Cleaning/normalization | 5 | 88.0% | WARNING | HIGH |
| Exact dedup | 5 | 95.0% | PASS | HIGH |
| Near dedup | 8 | 35.0% | FAIL | HIGH |
| Accounting | 4 | 99.0% | PASS | HIGH |
| Split correctness | 5 | 92.0% | PASS | HIGH |
| Leakage prevention | 8 | 35.0% | FAIL | HIGH |
| Tokenizer training correctness | 7 | 98.0% | PASS | HIGH |
| Tokenizer quality | 5 | 95.0% | PASS | HIGH |
| Final tokenizer freeze | 4 | 97.0% | PASS | HIGH |
| Token counting | 5 | 97.0% | PASS | HIGH |
| 50M subset correctness | 8 | 98.0% | PASS | HIGH |
| Reproducibility | 7 | 65.0% | WARNING | HIGH |
| Code correctness | 5 | 72.0% | WARNING | HIGH |
| Code cleanliness/humanization | 3 | 86.0% | PASS | MEDIUM |
| Performance/optimality | 3 | 80.0% | WARNING | MEDIUM |
| Test quality | 3 | 55.0% | FAIL | HIGH |
| Artifact consistency | 4 | 65.0% | FAIL | HIGH |
| Report/defense readiness | 3 | 70.0% | WARNING | HIGH |

Normalized weighted total: **78.0 / 100**.

## Findings by severity

### P1 — Critical

**AUD-001 — Large-bucket candidate recall and true-near leakage.** Star expansion measured 1.83% direct eligible-pair recall in the stratified sample and left 237 retained representative pairs across splits. This can bias future evaluation. Replace it with a recall-preserving deterministic strategy, add regression tests, and refreeze near-dedup and all downstream artifacts. Expensive focused rerun: yes. Must fix before M2/M3 and the final paper: yes.

### P2 — Major

**AUD-002 — Absolute-path portability.** Live manifest references point to OneDrive. Convert them to repository-relative or resolve them centrally, then regenerate dependent hashes. Expensive rerun: no. Must fix before M2/M3 if the repository will move.

**AUD-003 — Test adequacy.** The critical branch has no regression test. Add large-bucket recall, true-near leakage, cutoff, resume, and relocation cases before refreezing. Expensive rerun: no. Must fix before M2/M3.

**AUD-004 — Resume/write safety.** Band validation, atomic publication, rebuild containment, and stage-integrity guards are weaker than needed for recovery. Harden these before the required focused rerun. Expensive rerun: no for the code fix.

### P3 — Minor

- **AUD-005:** access date missing; per-source license/revision gaps are honestly marked unavailable.
- **AUD-006:** root/data READMEs and dedup/leakage documentation are stale.
- **AUD-007:** unknown-rate denominators and exact 50M cutoff need explicit metadata.
- **AUD-008:** resumed normalization-change counters are unavailable; band/intermediate evidence should be hash-bound.

### P4 — Optional

**AUD-009 — Maintainability polish.** Break up long report/handoff builders only when they next require substantive edits. A correct `tokenizer.json` may be added later if a verified compatible conversion path exists; no placeholder is needed.

## Missing / weak areas

- Recall-preserving large-bucket near-candidate generation
- A leakage definition that covers true near pairs, not only accepted edges
- Regression tests for the critical large-bucket branch
- Portable processed-data references
- Stronger resume and atomic-publication guarantees
- Explicit access date, unknown-rate denominators, component topology, and 50M cutoff metadata

## Must fix before M2/M3

1. Replace star expansion for observed large buckets with deterministic, recall-preserving enumeration or an equivalently validated method.
2. Add tests for large-bucket recall, omitted true-near leakage, connected-component semantics, and exact cutoff behavior.
3. Reuse only hash-verified normalization/exact/fingerprint/band state; rerun candidate verification, components, splits, and leakage checks.
4. Rebuild the tokenizer sample and retrain/refreeze tokenizer candidates and final 16K unless the corrected train sample is proven byte-identical. Then recount and refreeze the 50M sequence.
5. Make consumed data paths relocation-safe and record the exact 50M boundary in the handoff.
6. Regenerate handoff, reports, and DoD from corrected measured artifacts; rerun all tests and independent leakage checks.

## Can wait until final paper

- Recovering an access date if reliable evidence exists
- Polishing long report/handoff functions
- Adding a correct `tokenizer.json` converter, if downstream tooling actually needs it
- Additional presentation figures beyond the evidence-bearing tables listed above
- Reconstructing unavailable normalization transformation counters; disclosure is preferable to another raw scan

## Strongest M1 areas

1. Raw and critical-artifact hashes verify with no mismatch.
2. Corpus accounting is exact from raw rows through split manifests.
3. Tokenizer candidate provenance is train-only, shared, deterministic, and hash-frozen.
4. The final 16K tokenizer and all full-corpus token totals independently validate.
5. The 50M selection replay matches IDs, ordering, quotas, shortage redistribution, and exact cutoff.

## Weakest M1 areas

1. The large-bucket LSH shortcut sacrifices required recall.
2. Existing leakage checks only prove properties of already detected edges.
3. Absolute data paths prevent safe relocation as-is.
4. The test suite missed the exact branch that causes the scientific failure.
5. Completion documentation, resume safety, and some provenance/metric details are incomplete.

## Final DoD reassessment

**NOT COMPLETE.**

The old DoD is numerically consistent with its inputs, but its leakage and completion gates are based on an incomplete candidate graph. A frozen tokenizer and 50M manifest cannot make the current split research-ready once retained true-near cross-split pairs have been demonstrated.

## Final recommendation

Do not start M2/M3 on the current frozen split. Perform a focused correction beginning at large-bucket candidate generation; do not repeat raw normalization or exact deduplication. Refreeze every downstream artifact, make the path contract portable, and repeat the independent leakage gate. Only then should M1 be declared complete.

No Git or GitHub operation was performed.
