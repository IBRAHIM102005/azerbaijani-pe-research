# M1 Final Post-Repair Independent Audit

## Executive verdict

Overall M1 Quality: **95.8%**  
Scientific Readiness: **97%**  
Engineering Readiness: **94%**  
Reproducibility Readiness: **94%**  
M2/M3 Handoff Readiness: **97%**

Final verdict: **READY**  
Definition-of-Done reassessment: **COMPLETE WITH WARNINGS**

No P0, P1, or P2 issue remains. The original large-bucket candidate failure is closed: complete observed-bucket expansion, exact verification, regenerated downstream artifacts, layered leakage checks, and full row-reference validation all pass. The remaining items are reported limitations, not reasons to rerun M1 or delay M2/M3.

## What was independently verified

- The authoritative SQLite/report state contains 6,444,499 repaired candidates and 86,697 exact accepted edges.
- The all-large-bucket audit evaluated 3,503,354 pair opportunities across all 59 formerly problematic bucket events. It captured 321,263/321,263 eligible true pairs, with zero misses.
- Repaired manifests contain 5,574,885 train, 309,677 validation, and 309,369 test documents. Their SHA-256 values match the full validator.
- The final tokenizer and all three candidate audits match one repaired-train-only corpus. Unknown rates use explicit SentencePiece-token denominators.
- The document-token index covers 6,193,931 documents and 605,720,737 tokens.
- The 50M replay matches every selected ID and its order, plus the exact inside-document boundary.
- All 34 raw-path integrity checks pass, 30/30 tests pass, and the full processed-row-reference validator passes.

## Requirement compliance matrix

| Requirement | Implementation | Evidence | Measured result | Status |
| --- | --- | --- | --- | --- |
| Native core; translated source excluded | configs/frozen/m1.yaml | source_registry.yaml and handoff | Six native components included; translated-enwiki and unresolved bhos excluded | PASS |
| Conservative Azerbaijani-preserving cleaning | src/data/normalize.py | Tests and source accounting | NFC/case/letters preserved; removals reconciled | PASS |
| Exact global dedup | src/data/dedup.py | SQLite and exact report | 144,480 removals; 6,209,184 exact unique | PASS |
| High-recall near candidates and exact 0.95 check | src/data/near.py | Sample plus all-59-bucket audits | 6,444,499 verified candidates; 321,263/321,263 eligible large-bucket pairs captured | PASS |
| Cluster-aware 90/5/5 split | src/data/split.py | Manifest hashes and full validator | 5,574,885 / 309,677 / 309,369 | PASS |
| No cross-split duplicate leakage | src/data/leakage.py | Layered leakage artifacts | Zero known retained true-near cross-split pairs | PASS |
| 8K/16K/32K same train-only input | src/tokenizer | Training corpus and sample hashes | 1,000,000 repaired-train documents shared exactly | PASS |
| Final 16K BPE | src/tokenizer/train.py | Frozen model load and hashes | 16,000 pieces; special IDs stable | PASS |
| Real final-tokenizer counts | src/tokenizer/counts.py | 6.19M-row count index and validator | 605,720,737 tokens including one EOD/document | PASS |
| Deterministic 50M train sequence | src/data/sampling.py | Independent replay and subset hash | 277,027 unique train docs; exact boundary frozen | PASS |
| Raw immutability | scripts/verify_raw_immutability.py | raw_immutability.json | 34/34 path checks; zero mismatches | PASS |
| Portable handoff | src/data/paths.py | Relocation simulation | Operational paths resolve under alternate root | PASS |
| Final validation and tests | finalization scripts and tests | m1_validation.json and JUnit | Full validator pass; 30/30 tests | PASS |

## Data pipeline audit

The corrected accounting is 8,227,654 raw records, 1,873,990 too-short removals, 144,480 exact removals, 6,209,184 exact-unique documents, 15,253 near removals, and 6,193,931 retained documents. Both accounting identities reconcile. Cleaning remains conservative and preserves Azerbaijani orthography. The 50-letter filter's concentration in `mediocore-books` remains a reportable limitation.

## Exact dedup audit

Exact canonical-text groups use SHA-256 and a deterministic representative. The database and report agree on 144,480 removals and 6,209,184 unique documents. Provenance is retained through source, shard, and row identifiers.

## Near-dedup audit

### Large-bucket candidate recall

The repair uses complete, chunked unordered-pair enumeration in every observed colliding bucket. The largest observed bucket has 940 documents; star-expanded and skipped buckets are both zero. The repaired table contains 6,444,499 unique pairs and all were checked using direct character-5-gram Jaccard. Exactly 86,697 edges met the 0.95 threshold.

The stratified sample improved from 1.826753% prerepair recall to 100%. The stronger exhaustive audit covered all 59 formerly problematic bucket events: 321,263 eligible true pairs, 321,263 captured, zero missed, and zero retained cross-split true-near representative pairs.

### Connected-component transitivity

Accepted edges are direct Jaccard >= 0.95. Components are transitive and do not imply all-pairs similarity. The audit sampled the 25 largest nontrivial clusters; 24 had at least one sampled endpoint below 0.95. Minimum sampled endpoint similarity was 0.820, and minimum representative-to-member similarity was 0.858. This is more conservative than complete-link deduplication, is frozen before modeling, and is explicitly documented.

## Split and leakage audit

Observed split proportions are train 90.005604%, validation 4.999684%, and test 4.994712%. Document IDs, canonical hashes, cluster IDs, and accepted edges do not cross splits. The independent all-large-bucket audit finds no retained true-near cross-split pair. All 237 prerepair confirmed pairs are resolved.

## Tokenizer audit

All candidates saw the same first 1,000,000 document-ID-sorted repaired train documents. Training corpus SHA-256: `da0ff4b8209ab40e98afc96c71584a15defbd962d2f50e9b4f5ebc4e0a65a1d1`. Sample-manifest SHA-256: `937db7d8af5a744e55187d693129f5acc322f48066a432e1b6d887d97a525122`.

| Vocab | Tokens/word | Chars/token | UNK / token denominator | UNK rate | Model SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 8000 | 1.9212 | 4.0862 | 1 / 11,132,227 | 8.983e-08 | 52b4dd1fc9c0ae017bba13fd41aa020b05409356a397d840ca699b47428e4bbe |
| 16000 | 1.6743 | 4.6888 | 1 / 9,701,653 | 1.031e-07 | be05949c40afbe6031eee5678f49f5f49fad81cb1dd5fa8fb56c67f181222534 |
| 32000 | 1.5086 | 5.2038 | 1 / 8,741,364 | 1.144e-07 | 6cb625cdbb620ebc5e851da793edac26eea12c433a34c765616a22d5857a82fc |

The final model is SentencePiece BPE with 16,000 pieces, identity normalization, character coverage 1.0, byte fallback disabled, `<unk>` ID 0, `<eod>`/EOS ID 1, and BOS/PAD disabled. Azerbaijani-specific round trips pass. Candidate rates are sample rates; full-corpus reports give counts by source/split from the final 16K model and do not silently compare different denominators.

## Token-count audit

Train has 544,498,912 tokens, validation 30,329,083, and test 30,892,742, for 605,720,737 total including one `<eod>` per document.

| Source | All-split 16K tokens |
| --- | --- |
| anl-news | 237,559,904 |
| azwiki | 74,610,988 |
| elite-blogs | 1,528,244 |
| elite-books | 7,950,165 |
| eqanun | 64,643,900 |
| mediocore-books | 219,427,536 |

## 50M selection replay

The frozen sequence contains 277,027 unique train documents and 50,062,887 whole-document tokens. Blogs supply only 1,406,264 train tokens; the shortfall is recorded and redistributed without replacement.

| Group | Requested tokens | Selected tokens | Selected share |
| --- | --- | --- | --- |
| Blogs | 5,000,000 | 1,406,264 | 2.809% |
| Books | 15,000,000 | 16,121,567 | 32.203% |
| Laws | 7,500,000 | 8,347,290 | 16.674% |
| Native Wikipedia | 10,000,000 | 10,748,666 | 21.470% |
| News | 12,500,000 | 13,439,100 | 26.844% |

The exact 50M boundary is one-based sequence position 276,626, document `db05beef4e2c5dec4bf978a78afd788f579429852afea925852f7122b7608c36`. Cumulative tokens before it are 49,995,815; consume 4,185 of its 6,799 tokens and stop before `<eod>`. Subset SHA-256: `805f7b18c17007d2c11628419ddcf0afea7e615240358d5bfa7f57684c681d48`.

## Reproducibility audit

The handoff freezes seeds, package versions, config and artifact hashes, source registry, manifest hashes, tokenizer hashes, token index, and sequence hash. Candidate and verification stages use config-bound completion states. Partial artifacts use distinct names and are promoted only after validation. The prerepair database and reports remain preserved as historical evidence.

## Portability audit

**Safe to move: CONDITIONAL.** Repository-internal operational references are relative and passed a simulated move to `C:/Research/azerbaijani-positional-encoding`. Historical raw-inventory and audit evidence retain original access/display paths intentionally. After moving, keep the external DOLLMA clone at the configured relative location or set `AZ_PE_DOLLMA_ROOT` to it. M2/M3 consumption of processed data and tokenizer artifacts does not depend on the old OneDrive path.

## Code quality / humanization

Human-written quality: **89%**. Readability: **91%**. Maintainability: **90%**. Code correctness: **92%**.

The cleanest repair-related files are `src/data/near.py`, `src/data/paths.py`, `src/data/leakage.py`, `src/data/manifests.py`, and `tests/unit/test_near_repair.py`. Naming is generally concrete, comments explain scientific decisions, and the repair avoids decorative/tutorial prose. `generate_m1_handoff.py` is necessarily long, and `max_star_bucket` remains as a legacy compatibility name despite complete enumeration; neither affects results.

## Performance audit

Current performance quality: **88%**. The implementation streams Parquet, batches SQLite writes, uses a single controlled writer, and checkpoints by band and verification key. Complete expansion was practical for the observed maximum bucket of 940. Full artifact validation is deliberately I/O-heavy but is a final gate rather than a routine training dependency. No GPU redesign is warranted.

## Test adequacy

Test pass status: **100%** (30/30). Test adequacy: **90%**. The suite now covers the original non-anchor large-bucket failure, candidate completeness/deduplication/determinism, resume equivalence, connected components, independent omitted-pair leakage, portable paths, quota shortage, exact token boundary, incomplete-stage rejection, frozen tokenizer, frozen subset, and end-to-end synthetic flow. The 14 focused repair regressions also pass.

## Artifact and documentation consistency

The final validator reconciles all repaired counts, hashes, tokenizer artifacts, token records, 50M references, and exact boundary. `m1_handoff.json` and `m1_definition_of_done.json` agree with the repaired evidence. Prerepair files live under explicit historical paths. Root/data READMEs, analysis, and data card no longer claim that preprocessing has not begun and disclose the repair before model training.

## Scientific confidence

| Question | Answer | Evidence |
| --- | --- | --- |
| Is M1 scientifically valid for the PE comparison? | YES | All variants inherit one frozen corpus, tokenizer, sequence, and budget; repaired leakage gates pass. |
| Could known data leakage bias future PE results? | NO | No known ID, exact, accepted-edge, cluster, sampled, or all-large-bucket true-near pair crosses splits. |
| Could tokenizer construction favor one PE variant? | NO | One frozen 16K model is shared by all variants and was trained on train only. |
| Is the same data and order guaranteed? | YES | The subset manifest and data-seed-2026 order are hash-frozen and independent of model seeds. |
| Are low-data conditions reproducible? | YES | The exact 50M token boundary is machine-readable and independently replayed. |
| Is translated data excluded? | YES | translated-enwiki is absent from the core source map and all manifests. |
| Is dedup aggressive enough? | YES | Exact global dedup plus repaired high-recall LSH candidates and exact 0.95 verification are applied. |
| Is dedup too aggressive? | PARTIAL | Connected closure is intentionally conservative for leakage; sampled endpoints reach 0.8199 and this limitation is explicit. |
| Could the short filter distort source balance? | YES | It disproportionately removes mediocore-books fragments; source-aware token quotas and reporting make the effect visible. |
| Is 16K defensible? | YES | It is preregistered, technically valid, low-UNK, and audited beside 8K and 32K on identical train text. |
| Is the 50M corpus defensible? | YES | Train-only, unique, deterministic, source-aware, shortage-explicit, and exact-boundary replayed. |
| Can M2/M3 start without rerunning M1? | YES | The handoff exposes relative paths, hashes, tokenizer, manifests, counts, and stop semantics. |
| Can the repository move out of OneDrive? | PARTIAL | Internal artifacts relocate; set AZ_PE_DOLLMA_ROOT if the external DOLLMA clone is not adjacent at the configured relative path. |
| Are there absolute-path risks? | PARTIAL | Only provenance/display evidence retains original paths; operational paths are relative and relocation-tested. |
| Could an identified M1 flaw invalidate the paper later? | NO | No unresolved P0/P1/P2 remains; documented source metadata, short-filter, and transitivity limitations must be reported. |

## Weighted scorecard

The supplied historical weights total **105**, not 100. The normalized result is `sum(weight × score) / 105`.

| Area | Weight | Score | Weighted contribution | Status | Confidence |
| --- | --- | --- | --- | --- | --- |
| Research design compliance | 8 | 97.0 | 7.3905 | PASS | HIGH |
| Source/provenance | 5 | 88.0 | 4.1905 | WARNING | HIGH |
| Cleaning/normalization | 5 | 90.0 | 4.2857 | WARNING | HIGH |
| Exact dedup | 5 | 97.0 | 4.6190 | PASS | HIGH |
| Near dedup | 8 | 96.0 | 7.3143 | PASS | HIGH |
| Accounting | 4 | 100.0 | 3.8095 | PASS | HIGH |
| Split correctness | 5 | 98.0 | 4.6667 | PASS | HIGH |
| Leakage prevention | 8 | 99.0 | 7.5429 | PASS | HIGH |
| Tokenizer training correctness | 7 | 99.0 | 6.6000 | PASS | HIGH |
| Tokenizer quality | 5 | 97.0 | 4.6190 | PASS | HIGH |
| Final tokenizer freeze | 4 | 99.0 | 3.7714 | PASS | HIGH |
| Token counting | 5 | 99.0 | 4.7143 | PASS | HIGH |
| 50M subset correctness | 8 | 99.0 | 7.5429 | PASS | HIGH |
| Reproducibility | 7 | 94.0 | 6.2667 | PASS | HIGH |
| Code correctness | 5 | 92.0 | 4.3810 | PASS | HIGH |
| Code cleanliness/humanization | 3 | 89.0 | 2.5429 | PASS | MEDIUM |
| Performance/optimality | 3 | 88.0 | 2.5143 | PASS | MEDIUM |
| Test quality | 3 | 90.0 | 2.5714 | PASS | HIGH |
| Artifact consistency | 4 | 99.0 | 3.7714 | PASS | HIGH |
| Report/defense readiness | 3 | 95.0 | 2.7143 | PASS | HIGH |

## Findings by severity

There are no P0 blockers, P1 critical findings, or P2 major findings.

- **P3 — connected-component transitivity:** sampled endpoints reach 0.8199; preserve and report the already-frozen policy.
- **P3 — upstream provenance gaps:** per-source revisions/licenses remain unavailable and Books labels are qualified inferences.
- **P3 — short-filter distribution:** the 50-letter threshold disproportionately removes Books II fragments.
- **P4 — provenance paths:** original display paths remain in historical evidence; operational artifacts are portable.
- **P4 — legacy naming:** one compatibility parameter retains the former star terminology although repaired execution is complete-pair only.

## Previous issue closure

| Issue | Before | Fix | Validation | After | Status |
| --- | --- | --- | --- | --- | --- |
| P1-LSH | P1 | Star expansion replaced by complete chunked pair enumeration. | Sample and all 59 large buckets have 100% eligible true-pair recall. | resolved | CLOSED |
| P1-LEAKAGE-ASSURANCE | P1 | Layered graph plus independent large-bucket and former-pair audits. | Zero known retained cross-split true-near pairs; 237/237 resolved. | resolved | CLOSED |
| P2-PORTABILITY | P2 | Repository-relative operational references and centralized resolver. | Independent replay under C:/Research simulation passed. | P4 external-root condition | CLOSED |
| P2-TEST-COVERAGE | P2 | Large-bucket, resume, leakage, relocation, atomic-state, and boundary tests added. | 30/30 full suite and 14/14 focused repair set pass. | resolved | CLOSED |
| P2-RESUME-SAFETY | P2 | Config-bound stage states, durable checkpoints, partial names, atomic metadata, guarded promotion. | Resume-equivalence and incomplete-stage regressions pass. | resolved | CLOSED |
| P3-README | P3 | Root and data READMEs updated to repaired frozen M1 state. | No stale 'preprocessing not started' statement remains. | resolved | CLOSED |
| P3-ACCESS-DATE | P3 | Access date and its basis recorded in source registry and handoff. | 2026-08-28 is present with a local revalidation basis. | resolved | CLOSED |
| P3-UNK-DENOMINATOR | P3 | Every candidate rate records count, token denominator, and policy; full-corpus values remain counts by split/source. | Handoff and analysis table expose denominators explicitly. | resolved | CLOSED |
| P3-50M-BOUNDARY | P3 | Exact inside-document stopping structure frozen and replayed. | Boundary ID, position, cumulative count, 4,185 consumed tokens, and pre-EOD stop agree. | resolved | CLOSED |
| P3-TRANSITIVITY-DOC | P3 | Edge-versus-component semantics and measured transitivity sample documented. | Near report, data card, analysis, and handoff agree. | P3 documented limitation | ACCEPTED |

## Missing / weak areas

Upstream component-level license/revision detail is still unavailable. The minimum-length filter and transitive cluster closure require explicit limitations text. Historical provenance artifacts intentionally record the machine on which raw access was verified. These do not invalidate the frozen M1 handoff.

## Must fix before M2/M3

Nothing. No unresolved P0, P1, or research-invalidating P2 remains.

## Can wait until final paper

Explain the short-text distribution effect and connected-component transitivity, retain qualified source-provenance language, and cite the repaired audit evidence in the methods/reproducibility material.

## Strongest M1 areas

The strongest areas are exact accounting, repaired near-candidate recall, layered leakage validation, train-only tokenizer provenance, independently replayed 50M order/boundary, raw immutability, and full row-reference validation.

## Weakest M1 areas

The remaining weak points are upstream metadata completeness, the domain effect of the frozen short-text filter, the aggressiveness of connected-component closure, and some long finalization scripts. These are documented and non-blocking.

## Final DoD reassessment

**COMPLETE WITH WARNINGS.** Every repaired hard gate passes and `complete=true` is supported by current artifacts. The warnings are methodological/provenance limitations, not incomplete computation.

## Final recommendation

M2/M3 may start from `data/metadata/m1_handoff.json`. Do not rerun M1. All positional-encoding conditions must use the frozen tokenizer, manifests, sequence order, and exact 50M stop semantics recorded there.
