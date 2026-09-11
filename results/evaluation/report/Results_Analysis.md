# M4 Results and Analysis

## Evaluation protocol

We evaluated five positional encoding methods across model seeds 17, 42, 1234, 2027 and 5003. Each model was trained by M3 for 50 million tokens. Validation uses the 5M/10M/20M/50M milestones; the held-out test uses only the final 50M models. Exact token counts for intermediate checkpoints are 5,046,272, 10,027,008 and 20,054,016. The validation evidence, inference environment and evaluation protocol were frozen before the final test batch began. The concrete window and statistical rules in this M4 package were specified for this evaluation; an earlier external preregistration is not asserted.

The frozen M1 SentencePiece tokenizer projects newlines to spaces and appends one EOD token per document, without BOS. Documents are evaluated independently. At context 512, overlapping windows with stride 256 score each token after the first exactly once, including EOD. Each run's NLL is the sum of target losses divided by the number of scored targets. The headline value averages these token-weighted NLLs across the five seeds; perplexity is exp(mean NLL). The separate mean of per-seed perplexities is available in primary.csv.

## Primary held-out results

| PE | Mean NLL | Seed SD | exp(mean NLL) | Conditional 95% CI |
|---|---:|---:|---:|---|
| learned | 5.793908 | 0.008968 | 328.2936 | [5.777071, 5.811450] |
| sinusoidal | 6.894045 | 0.026865 | 986.3828 | [6.883096, 6.906414] |
| rope | 5.532893 | 0.006336 | 252.8745 | [5.513368, 5.554253] |
| alibi | 5.434068 | 0.006519 | 229.0793 | [5.413164, 5.458267] |
| nope | 5.648448 | 0.005536 | 283.8505 | [5.629284, 5.667937] |

The primary test contains 309,369 documents and 30,583,373 scored targets per seed. The lowest observed mean NLL is 5.434068 for alibi, corresponding to perplexity 229.0793. alibi meets the prespecified M4 comparison rule against all four alternatives.

Uncertainty uses 10,000 document bootstrap draws with bootstrap seed 5, shared across all methods and model seeds. For each draw, document loss sums and token counts are resampled together. The 95% percentile intervals are conditional on the five trained seeds; sample SD across seeds is reported separately. Approximate two-sided null-centered bootstrap p-values use a plus-one correction and Holm adjustment across all ten pairwise comparisons. The comparison rule requires a CI excluding zero, Holm p<0.05, consistent direction in at least four of five seeds, and an observed absolute difference of at least 0.01 NLL. This practical threshold is carried forward from the supplied five-seed remediation plan and frozen before final test. It applies to the point estimate; it is not an equivalence test or a claim that the entire CI exceeds 0.01.

The canonical best-minus-runner contrast is alibi − rope: ΔNLL -0.098825, 95% conditional CI [-0.100657, -0.095834], Holm p=0.001000; 5/5 seed directions agree. The corresponding relative change in exp(mean NLL) is -9.410%. All four best-minus-comparator contrasts are in best_vs_others.csv. Tied-lowest counts across the five seeds are: learned: 0/5; sinusoidal: 0/5; rope: 0/5; alibi: 5/5; nope: 0/5; these are descriptive, with ties retained.

## Explicit ablation: RoPE versus NoPE

The preregistered ablation in the supplied master plan removes RoPE from the shared architecture and compares it with NoPE under the same data order and matched seeds. Its canonical RoPE-minus-NoPE effect is -0.115554 NLL, 95% conditional CI [-0.117861, -0.112598], Holm p=0.001000, relative PPL change -10.913%. Directional consistency is 5/5; the practical-effect flag is True; the combined comparison rule is True. Negative Δ favors RoPE; positive Δ favors NoPE. This is one of the existing ten comparisons, not a separate uncorrected significance test. Exact values are in ablation_rope_nope.csv.

## Learning curves and context extension

Figure 4 shows validation learning curves against actual training-token counts. Figure 5 shows final primary test scores. Figure 6 evaluates all documents with at least 2048 tokens, including EOD, and scores the identical final 256 targets of each document's first 2048-token prefix at contexts 512, 1024 and 2048. Differences therefore concern available context on matched targets; they do not mix different target populations. This is a narrower estimand than the all-document headline evaluation. Learned positional embeddings are reported as not applicable above 512, without adding parameters or changing the trained model. Figure 7 presents all pairwise contrasts. Exact learning and length results are in learning.csv and length.csv.

Lowest observed validation mean by milestone: 10m: alibi; 20m: alibi; 50m: alibi; 5m: alibi. A change in the leading method across those budgets is False. This describes the same training trajectories, not separately trained short-budget models. sample_efficiency.csv reports mean/SD, observed ranks and change from the 5M checkpoint. No absolute target NLL was supplied in the plans, so a target-reaching token count or interpolated threshold is not invented. Budget-dependent rankings and endpoint rankings are interpreted separately.

## Limits and reproducibility

Results concern this frozen corpus, architecture, tokenizer, compute budget and five initialization seeds. Document-level intervals do not include uncertainty over new training seeds or correlated source/book clusters. Same-runtime deterministic settings are recorded; bitwise equivalence across different hardware/software is not claimed. M3 recorded a dirty working tree. The uploaded model/training sources match the recorded training commit, but uncommitted server changes cannot be independently reconstructed from that commit alone. Checkpoint hashes, strict state-dictionary loading and payload checks tie every evaluated model to the supplied handoff. Interrupted jobs can resume; committed test jobs are verified and skipped.

Final test manifest SHA-256: `aa58372d3412cf5bbfcffb14e67729e3ca0e5ebc378ed0b5b614bb8c43131368`.
