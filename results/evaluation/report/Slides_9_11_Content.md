# Slides 9–11: content and speaker notes

## Slide 9 — Final test and statistical comparison
- Lowest observed mean: alibi, NLL 5.4341, PPL 229.08.
- 10,000 shared document bootstrap draws; Holm correction for ten pairs.
- alibi meets the prespecified M4 comparison rule against all four alternatives.
- Visuals: Figure 5; use Figure 7 for pairwise evidence.

Speaker note: Separate observed rank from supported contrasts. CIs condition on the trained seeds; seed SD measures variation across those five runs. The combined rule also requires an observed absolute effect of at least 0.01 NLL. A p-value alone does not establish practical importance.

## Slide 10 — Sample efficiency
- Same training trajectories evaluated at 5M/10M/20M/50M milestones.
- Lowest observed validation means: 10m: alibi; 20m: alibi; 50m: alibi; 5m: alibi.
- Leading method changes across budgets: False.
- Visual: Figure 4; exact values in sample_efficiency.csv.

Speaker note: Explain actual optimizer-boundary token counts and mean±seed SD. These are descriptive learning curves from the same trajectories. No absolute target NLL or interpolated crossing time was invented. Distinguish early-budget behavior from the final test endpoint.

## Slide 11 — Context extension and limitations
- Contexts 512/1024/2048 score the same 256 targets in long documents.
- Learned >512: not applicable.
- RoPE−NoPE ablation: ΔNLL -0.1156; Holm p=0.0010; combined rule=True.
- Findings are specific to this data, model size, tokenizer and 50M budget.
- Visual: Figure 6.

Speaker note: The length cohort is different from the headline cohort. Discuss the measured pattern from length.csv without comparing its absolute NLL directly to Figure 5. Document bootstrap does not capture source-cluster or new-seed uncertainty.
