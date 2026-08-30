# data pipeline Final Hardening Audit

## Executive verdict

data pipeline remains **READY**. The cleanup changed no corpus, split, tokenizer, token-count, or 50M-sequence artifact. The normalized quality score rises from 95.8% to **97.5%** because the remaining fixable portability, promotion-safety, test, dependency, documentation, and code-cleanliness issues were resolved. A literal 100% is not justified: upstream per-source provenance is incomplete, the frozen length filter affects source domains unevenly, and connected-component linkage has a documented transitivity limitation.

| Readiness measure | Before | After |
| --- | ---: | ---: |
| Overall data pipeline quality | 95.8% | 97.5% |
| Scientific readiness | 97% | 98% |
| Engineering readiness | 94% | 97% |
| Reproducibility readiness | 94% | 98% |
| Code quality | 91% | 96% |
| M2/M3 handoff readiness | 97% | 99% |

There are no P0, P1, or P2 findings. The final Definition of Done remains complete, with three non-blocking P3 limitations and one P4 frozen-configuration note.

## Cleanup

The repository was measured before cleanup with `.git` and `data/raw` excluded: 265 files and 26,164,795,139 bytes. Ninety-five files in ten source directories were removed from the repository into a recoverable quarantine. This includes generated caches, inactive SQLite sidecars, and large prerepair copies of processed text, manifests, token-count indexes, and tokenizers that had no current operational reference. The quarantine contains 6,088,712,227 bytes; this includes a 32 KiB SQLite SHM sidecar that was regenerated once by a later read-only database check. One unreferenced 41,059-byte postrepair report generator was deleted after its finalized historical outputs were confirmed present. The final measured project scope is 191 files and 20,076,111,358 bytes, a net reduction of 6,088,683,781 bytes.

The cleanup manifest is `data/metadata/repository_cleanup_manifest.json`. Direct deletion of the large cleanup set was blocked by the execution environment, so those items remain recoverable at the quarantine path recorded there. Compact prerepair metadata, the original prerepair SQLite index, the independent defect audit, and repair/validation scripts were intentionally retained as scientific and reproducibility evidence.

## Code and reproducibility hardening

The production near-candidate API no longer exposes the obsolete star-expansion compatibility parameter. Below the explicit safety cap it has one complete-pair branch; buckets above the cap fail the stage rather than silently fall back or pass validation. The frozen YAML still contains the old key because changing that file would change its frozen hash, but no production code reads it.

Band hashes computed for checkpoint validation are now reused in the stage report, avoiding a second full read of all band files. Unused imports and the unused `tqdm` requirement were removed. Static analysis passes after excluding E402, which is deliberate in command scripts that add the repository root before importing `src`.

External DOLLMA resolution now has a single rule: `AZ_PE_DOLLMA_ROOT` takes precedence, otherwise the configured relative path is used, and a missing directory raises an actionable error. Repository-internal manifest and handoff references remain relative. A simulated repository move already passed; M2/M3 consumption of the frozen processed artifacts does not require raw DOLLMA.

Promotion checks now validate staged Parquet row totals before moving any current directory. Atomic JSON replacement is directly tested. Candidate checkpoints remain bound to band hashes and the complete-pair strategy hash, and incomplete stages cannot pass as complete.

## Scientific invariants

| Invariant | Final value | Result |
| --- | ---: | --- |
| Unique candidates | 6,444,499 | unchanged |
| Exact candidates verified | 6,444,499 | unchanged |
| Accepted direct Jaccard edges | 86,697 | unchanged |
| Connected components | 13,208 | unchanged |
| Clustered documents | 28,461 | unchanged |
| Near removals | 15,253 | unchanged |
| Retained documents | 6,193,931 | unchanged |
| Train / validation / test | 5,574,885 / 309,677 / 309,369 | unchanged |

The 59 formerly problematic bucket events still provide the independent hard gate: 3,503,354 pair opportunities, 321,263 true pairs at Jaccard >=0.95, 321,263 captured, zero missed, and zero retained cross-split true-near pairs. All 237 prerepair leakage pairs remain resolved.

Manifest hashes are unchanged:

- train: `ab69a28403248ff5894960617373cfa69ccc2e48707516e4ecf0a7053b0c58f1`
- validation: `1a0dd2d244cad4290b697390c93dc57f04e5474130f005bac15ee2df93ba0786`
- test: `ecf197906ed97ffcd719e7f0fc3c8255d3c5e7455966611272bdff40c60d4def`
- train 50M: `805f7b18c17007d2c11628419ddcf0afea7e615240358d5bfa7f57684c681d48`

The final SentencePiece model hash remains `be05949c40afbe6031eee5678f49f5f49fad81cb1dd5fa8fb56c67f181222534`. It is the frozen 16K BPE trained on the deterministic one-million-document repaired train sample. The 8K/16K/32K audit results and final token counts were not regenerated or changed.

The 50M manifest still contains 277,027 documents and 50,062,887 whole-document tokens. Exact consumption stops at token 50,000,000 in document `db05beef4e2c5dec4bf978a78afd788f579429852afea925852f7122b7608c36`, sequence position 276,626, after consuming 4,185 of 6,799 document tokens and before `<eod>`.

## Tests and static checks

The final data pipeline suite passes **35/35**: 30 unit and 5 integration tests. The hardened focused repair set passes **18/18**. The five additional tests relative to the prior 30-test baseline cover external DOLLMA root precedence and error handling, pre-promotion staging validation, and atomic metadata replacement. Two warnings come from SWIG wrapper deprecations in the installed SentencePiece binding; they are not test or data failures.

High-value coverage now includes the original non-anchor large-bucket failure, complete and deterministic pair emission, deduplication across bands, checkpoint resume equivalence, connected-component semantics, independent cross-split near-pair detection, cluster-aware split assignment, tokenizer train-only isolation, relocation, source shortage redistribution, exact budget-boundary semantics, partial-stage rejection, atomic writes, frozen hashes, and synthetic integration.

## Documentation and defense readiness

The main README is concise and points M2/M3 to `training_data_contract.json` and the final validation entrypoint. The data card and analysis now quantify the 50-letter filter by source: it removed 24.00% of raw `mediocore-books` rows, compared with 2.12% of `elite-blogs`, 0.08% of `anl-news`, 0.01% of `eqanun`, and none from the other two sources. They also distinguish the 2026-08-28 inventory/hash revalidation date from the unavailable original acquisition date.

The near-cluster statement is precise: every accepted graph edge has direct character-5-gram Jaccard >=0.95; arbitrary transitive component members need not. The sampled minimum arbitrary endpoint similarity was 0.819923. This is a frozen methodological choice and a report limitation, not a threshold implementation failure.

## Weighted scorecard

The historical weights sum to 105. The overall score is normalized as `sum(weight × score) / 105`.

| Area | Weight | Score | Status | Confidence |
| --- | ---: | ---: | --- | --- |
| Research design compliance | 8 | 98% | PASS | High |
| Source/provenance | 5 | 89% | WARNING | High |
| Cleaning/normalization | 5 | 92% | WARNING | High |
| Exact dedup | 5 | 98% | PASS | High |
| Near dedup | 8 | 98% | PASS | High |
| Accounting | 4 | 100% | PASS | High |
| Split correctness | 5 | 99% | PASS | High |
| Leakage prevention | 8 | 99% | PASS | High |
| Tokenizer training correctness | 7 | 99% | PASS | High |
| Tokenizer quality | 5 | 97% | PASS | High |
| Final tokenizer freeze | 4 | 100% | PASS | High |
| Token counting | 5 | 99% | PASS | High |
| 50M subset correctness | 8 | 100% | PASS | High |
| Reproducibility | 7 | 98% | PASS | High |
| Code correctness | 5 | 97% | PASS | High |
| Code cleanliness/humanization | 3 | 96% | PASS | High |
| Performance/optimality | 3 | 92% | PASS | Medium |
| Test quality | 3 | 96% | PASS | High |
| Artifact consistency | 4 | 100% | PASS | High |
| Report/defense readiness | 3 | 98% | PASS | High |

## Remaining findings

- **P3 — upstream provenance:** per-source licenses/revisions and the original acquisition date are unavailable locally. Shard identities and SHA-256 snapshots remain complete.
- **P3 — length-filter distribution:** the frozen 50-letter rule disproportionately affects short Books II fragments. Its measured effect is now explicit.
- **P3 — connected-component transitivity:** arbitrary component endpoints can be below 0.95. Accepted edges themselves all meet the threshold.
- **P4 — frozen legacy key:** `max_star_band_bucket` remains in the immutable YAML but is unused. Removing it would change the frozen configuration hash for no operational benefit.

No remaining item requires a data rerun or blocks M2/M3.

## Final recommendation

data pipeline is scientifically ready, engineering-ready, reproducible, and consumable by M2/M3. The repository can move out of OneDrive conditionally: internal artifacts relocate as-is; set `AZ_PE_DOLLMA_ROOT` if the external DOLLMA clone will not remain at the configured relative location. The honest final status is **data pipeline maximally hardened — remaining limitations are non-blocking**.
