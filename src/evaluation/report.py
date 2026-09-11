"""Publication figures and text generated exclusively from verified Evaluation scores."""
from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
from .common import PES,SEEDS,require,read_json,write_json,sha,atomic_file,verify_evidence
from .inputs import load_workspace
from .statistics import final_evidence

COLORS=dict(zip(PES,['#9356A0','#D57A2B','#237BB5','#2F927A','#6C7380']))

def analysis(work):
    work=Path(work)
    a=read_json(work/'tables/analysis.json')
    require(a['final_manifest_sha256']==sha(work/'final_test_manifest.json'),'Analysis belongs to another final test')
    return a

def figures(work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(work);a=analysis(work);out=work/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#253341','text.color':'#253341',
                         'axes.titlesize':13,'savefig.dpi':220,'pdf.fonttype':42,'svg.fonttype':'none'})
    captions={}
    def save(fig,name,caption):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
        plt.close(fig);captions[name]=caption
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    for pe in PES:
        rows=[r for r in a['learning'] if r['pe']==pe]
        x=np.array([r['actual_tokens']/1e6 for r in rows]);y=np.array([r['mean_nll'] for r in rows]);sd=np.array([r['seed_sd_nll'] for r in rows])
        ax.plot(x,y,'o-',color=COLORS[pe],label=pe,linewidth=1.7,markersize=4)
        ax.fill_between(x,y-sd,y+sd,color=COLORS[pe],alpha=.12)
    ax.set(xscale='log',xlabel='Actual training tokens (millions)',ylabel='Validation NLL (nats / target token)',
           title='Figure 4 · Validation learning curves')
    ax.set_xticks([5.046272,10.027008,20.054016,50],['5.046','10.027','20.054','50'])
    ax.grid(axis='y',alpha=.17);ax.legend(ncol=3,frameon=False)
    save(fig,'figure_4_learning','All validation documents; means and ±1 sample SD across five model seeds. Actual optimizer-boundary token counts are used. Vertical scale is cropped for comparison.')
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for i,r in enumerate(a['primary']):
        vals=np.array([r[f'nll_seed_{s}'] for s in SEEDS])
        ax.scatter(i+np.linspace(-.12,.12,5),vals,s=26,color=COLORS[r['pe']],alpha=.65,zorder=3)
        ax.vlines(i,r['conditional_ci_low'],r['conditional_ci_high'],color='#203444',linewidth=2.5,zorder=4)
        ax.scatter([i],[r['mean_nll']],marker='D',s=34,c='#203444',zorder=5)
    ax.set_xticks(range(5),PES);ax.set(ylabel='Test NLL (nats / target token)',title='Figure 5 · Final models at context 512')
    ax.grid(axis='y',alpha=.17)
    save(fig,'figure_5_primary','Colored dots: five trained seeds. Diamond: mean token-weighted NLL. Dark intervals: 95% paired-document percentile bootstrap intervals conditional on those fixed seeds (10,000 draws). These intervals do not measure uncertainty over a population of model seeds. Vertical scale is cropped.')
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for pe in PES:
        rows=[r for r in a['length'] if r['split']=='test' and r['pe']==pe and r['status']=='complete']
        ax.errorbar([r['context'] for r in rows],[r['mean_nll'] for r in rows],
                    yerr=[r['seed_sd_nll'] for r in rows],fmt='o-',capsize=3,color=COLORS[pe],label=pe)
    ax.set_xticks([512,1024,2048]);ax.set(xlabel='Available context window',ylabel='Matched-target test NLL',
                                         title='Figure 6 · Context extension on identical targets')
    ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.17)
    ax.text(.99,.98,'Learned: >512 not applicable',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    save(fig,'figure_6_length','Only documents with at least 2048 tokens including EOD. Score the same target indices 1792–2047 of each document prefix at contexts 512, 1024 and 2048. Error bars: ±1 sample SD across five seeds. Learned >512 is undefined. This cohort/target set differs from Figure 5; vertical scale is cropped.')
    fig,ax=plt.subplots(figsize=(9,5.7))
    pairs=a['pairwise'];labels=[]
    for i,r in enumerate(pairs):
        color='#237B67' if r['supported_direction'] else '#77828E'
        ax.hlines(i,r['ci_low'],r['ci_high'],color=color,linewidth=2)
        ax.scatter(r['delta_nll_a_minus_b'],i,c=color,s=32,zorder=3)
        labels.append(r['a']+' − '+r['b'])
        ax.text(1.025,i,f"{r['p_holm']:.4f}  |  {r['consistent_seeds']}/5",transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.axvline(0,color='#263744',ls='--',lw=1);ax.set_yticks(range(10),labels);ax.invert_yaxis()
    ax.set(xlabel='Δ NLL (A − B); negative favors A',title='Figure 7 · Paired method comparisons')
    ax.text(1.025,1.02,'Holm p | direction',transform=ax.transAxes,fontsize=9)
    ax.grid(axis='x',alpha=.15);fig.subplots_adjust(right=.73)
    save(fig,'figure_7_pairwise','Mean NLL differences with 95% paired-document percentile CIs conditional on five seeds. Holm adjustment covers all ten null-centered bootstrap tests. Green: CI excludes zero, Holm p<0.05, direction agrees in at least four of five seeds, and observed absolute mean difference is at least 0.01 NLL. The practical flag applies to the point estimate; it does not assert the whole CI exceeds the margin.')
    write_json(out/'captions.json',captions)
    print(f'Figures 4–7 saved in {out}',flush=True)

def report(work):
    work=Path(work);a=analysis(work);out=work/'report';out.mkdir(exist_ok=True)
    best=next(r for r in a['primary'] if r['pe']==a['lowest_observed_mean_nll'])
    supported=a['supported_winner_against_all_four']
    canonical=a['best_vs_runner'];ablation=a['ablation_rope_nope']
    curve_description='; '.join(m+': '+', '.join(pe) for m,pe in a['trajectory_summary']['lowest_observed_methods_by_milestone'].items())
    rank_description='; '.join(pe+': '+str(sum(r['tied_for_lowest'] for r in a['seed_ranks'] if r['pe']==pe))+'/5' for pe in PES)
    conclusion=(f"{supported} meets the prespecified Evaluation comparison rule against all four alternatives."
                if supported else 'No method meets the Evaluation comparison rule against all four alternatives; the ordering of sample means alone does not establish a unique winner.')
    table='| PE | Mean NLL | Seed SD | exp(mean NLL) | Conditional 95% CI |\n|---|---:|---:|---:|---|\n'
    for r in a['primary']:
        table+=f"| {r['pe']} | {r['mean_nll']:.6f} | {r['seed_sd_nll']:.6f} | {r['ppl_exp_mean_nll']:.4f} | [{r['conditional_ci_low']:.6f}, {r['conditional_ci_high']:.6f}] |\n"
    text=f'''# Evaluation Results and Analysis

## Evaluation protocol

We evaluated five positional encoding methods across model seeds 17, 42, 1234, 2027 and 5003. Each model was trained by M3 for 50 million tokens. Validation uses the 5M/10M/20M/50M milestones; the held-out test uses only the final 50M models. Exact token counts for intermediate checkpoints are 5,046,272, 10,027,008 and 20,054,016. The validation evidence, inference environment and evaluation protocol were frozen before the final test batch began. The concrete window and statistical rules in this Evaluation package were specified for this evaluation; an earlier external preregistration is not asserted.

The frozen M1 SentencePiece tokenizer projects newlines to spaces and appends one EOD token per document, without BOS. Documents are evaluated independently. At context 512, overlapping windows with stride 256 score each token after the first exactly once, including EOD. Each run's NLL is the sum of target losses divided by the number of scored targets. The headline value averages these token-weighted NLLs across the five seeds; perplexity is exp(mean NLL). The separate mean of per-seed perplexities is available in primary.csv.

## Primary held-out results

{table}
The primary test contains {best['documents']:,} documents and {best['targets_per_seed']:,} scored targets per seed. The lowest observed mean NLL is {best['mean_nll']:.6f} for {best['pe']}, corresponding to perplexity {best['ppl_exp_mean_nll']:.4f}. {conclusion}

Uncertainty uses 10,000 document bootstrap draws with bootstrap seed 5, shared across all methods and model seeds. For each draw, document loss sums and token counts are resampled together. The 95% percentile intervals are conditional on the five trained seeds; sample SD across seeds is reported separately. Approximate two-sided null-centered bootstrap p-values use a plus-one correction and Holm adjustment across all ten pairwise comparisons. The comparison rule requires a CI excluding zero, Holm p<0.05, consistent direction in at least four of five seeds, and an observed absolute difference of at least 0.01 NLL. This practical threshold is carried forward from the supplied five-seed remediation plan and frozen before final test. It applies to the point estimate; it is not an equivalence test or a claim that the entire CI exceeds 0.01.

The canonical best-minus-runner contrast is {canonical['a']} − {canonical['b']}: ΔNLL {canonical['delta_nll_a_minus_b']:.6f}, 95% conditional CI [{canonical['ci_low']:.6f}, {canonical['ci_high']:.6f}], Holm p={canonical['p_holm']:.6f}; {canonical['consistent_seeds']}/5 seed directions agree. The corresponding relative change in exp(mean NLL) is {canonical['relative_ppl_change_a_vs_b_percent']:.3f}%. All four best-minus-comparator contrasts are in best_vs_others.csv. Tied-lowest counts across the five seeds are: {rank_description}; these are descriptive, with ties retained.

## Explicit ablation: RoPE versus NoPE

The preregistered ablation in the supplied master plan removes RoPE from the shared architecture and compares it with NoPE under the same data order and matched seeds. Its canonical RoPE-minus-NoPE effect is {ablation['delta_nll_a_minus_b']:.6f} NLL, 95% conditional CI [{ablation['ci_low']:.6f}, {ablation['ci_high']:.6f}], Holm p={ablation['p_holm']:.6f}, relative PPL change {ablation['relative_ppl_change_a_vs_b_percent']:.3f}%. Directional consistency is {ablation['consistent_seeds']}/5; the practical-effect flag is {ablation['passes_practical_effect']}; the combined comparison rule is {ablation['supported_direction']}. Negative Δ favors RoPE; positive Δ favors NoPE. This is one of the existing ten comparisons, not a separate uncorrected significance test. Exact values are in ablation_rope_nope.csv.

## Learning curves and context extension

Figure 4 shows validation learning curves against actual training-token counts. Figure 5 shows final primary test scores. Figure 6 evaluates all documents with at least 2048 tokens, including EOD, and scores the identical final 256 targets of each document's first 2048-token prefix at contexts 512, 1024 and 2048. Differences therefore concern available context on matched targets; they do not mix different target populations. This is a narrower estimand than the all-document headline evaluation. Learned positional embeddings are reported as not applicable above 512, without adding parameters or changing the trained model. Figure 7 presents all pairwise contrasts. Exact learning and length results are in learning.csv and length.csv.

Lowest observed validation mean by milestone: {curve_description}. A change in the leading method across those budgets is {a['trajectory_summary']['leading_method_changes_with_budget']}. This describes the same training trajectories, not separately trained short-budget models. sample_efficiency.csv reports mean/SD, observed ranks and change from the 5M checkpoint. No absolute target NLL was supplied in the plans, so a target-reaching token count or interpolated threshold is not invented. Budget-dependent rankings and endpoint rankings are interpreted separately.

## Limits and reproducibility

Results concern this frozen corpus, architecture, tokenizer, compute budget and five initialization seeds. Document-level intervals do not include uncertainty over new training seeds or correlated source/book clusters. Same-runtime deterministic settings are recorded; bitwise equivalence across different hardware/software is not claimed. M3 recorded a dirty working tree. The uploaded model/training sources match the recorded training commit, but uncommitted server changes cannot be independently reconstructed from that commit alone. Checkpoint hashes, strict state-dictionary loading and payload checks tie every evaluated model to the supplied handoff. Interrupted jobs can resume; committed test jobs are verified and skipped.

Final test manifest SHA-256: `{a['final_manifest_sha256']}`.
'''
    with atomic_file(out/'Results_Analysis.md','w') as f:f.write(text)
    latex=['\\begin{tabular}{lrrr}','\\hline','PE & Mean NLL & Seed SD & PPL \\\\','\\hline']
    for r in a['primary']:latex.append(f"{r['pe']} & {r['mean_nll']:.4f} & {r['seed_sd_nll']:.4f} & {r['ppl_exp_mean_nll']:.2f} \\\\")
    latex+=['\\hline','\\end{tabular}']
    with atomic_file(out/'primary_table.tex','w') as f:f.write('\n'.join(latex)+'\n')
    slides=f'''# Slides 9–11: content and speaker notes

## Slide 9 — Final test and statistical comparison
- Lowest observed mean: {best['pe']}, NLL {best['mean_nll']:.4f}, PPL {best['ppl_exp_mean_nll']:.2f}.
- 10,000 shared document bootstrap draws; Holm correction for ten pairs.
- {conclusion}
- Visuals: Figure 5; use Figure 7 for pairwise evidence.

Speaker note: Separate observed rank from supported contrasts. CIs condition on the trained seeds; seed SD measures variation across those five runs. The combined rule also requires an observed absolute effect of at least 0.01 NLL. A p-value alone does not establish practical importance.

## Slide 10 — Sample efficiency
- Same training trajectories evaluated at 5M/10M/20M/50M milestones.
- Lowest observed validation means: {curve_description}.
- Leading method changes across budgets: {a['trajectory_summary']['leading_method_changes_with_budget']}.
- Visual: Figure 4; exact values in sample_efficiency.csv.

Speaker note: Explain actual optimizer-boundary token counts and mean±seed SD. These are descriptive learning curves from the same trajectories. No absolute target NLL or interpolated crossing time was invented. Distinguish early-budget behavior from the final test endpoint.

## Slide 11 — Context extension and limitations
- Contexts 512/1024/2048 score the same 256 targets in long documents.
- Learned >512: not applicable.
- RoPE−NoPE ablation: ΔNLL {ablation['delta_nll_a_minus_b']:.4f}; Holm p={ablation['p_holm']:.4f}; combined rule={ablation['supported_direction']}.
- Findings are specific to this data, model size, tokenizer and 50M budget.
- Visual: Figure 6.

Speaker note: The length cohort is different from the headline cohort. Discuss the measured pattern from length.csv without comparing its absolute NLL directly to Figure 5. Document bootstrap does not capture source-cluster or new-seed uncertainty.
'''
    with atomic_file(out/'Slides_9_11_Content.md','w') as f:f.write(slides)
    print(f'Report and slide content saved in {out}',flush=True)

def export_results(work):
    work=Path(work);w=load_workspace(work);final_evidence(work,w);analysis(work)
    required=['tables/analysis.json','figures/figure_7_pairwise.png','report/Results_Analysis.md']
    require(all((work/n).exists() for n in required),'Run analyze, figures and report before export')
    common=[]
    for directory in ['tables','figures','report']:
        common.extend(p for p in (work/directory).rglob('*') if p.is_file())
    common += [work/n for n in ['workspace.json','runtime.json','smoke.json','validation_complete.json',
                                'validation_lock.json','final_test_state.json','final_test_manifest.json','events.jsonl'] if (work/n).exists()]
    common += list((work/'cache').glob('*/metadata.json'))
    common += list((work/'scores').rglob('*.json'))
    for name,include_raw in [('EVALUATION_REPORT.zip',False),('EVALUATION_RESULTS.zip',True)]:
        files=common+(list((work/'scores/test').rglob('*.npz')) if include_raw else [])
        with atomic_file(work/name) as f:
            with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in sorted(set(files)):z.write(p,'EVALUATION_WORK/'+str(p.relative_to(work)))
                z.writestr('README.txt','Contains real completed Evaluation evaluation outputs. No model weights or corpus text.\n'+
                           ('Use this result directory with the same Evaluation source package: analyze can rerun the document bootstrap on CPU. Validation raw arrays are omitted; only validation summaries are included.\n' if include_raw else
                            'Small review export: figures, tables, report and evidence summaries. Raw per-document arrays omitted; this ZIP alone cannot rerun bootstrap. Keep EVALUATION_RESULTS.zip for reproducibility.\n'))
        print(f'Exported {work/name} ({(work/name).stat().st_size/1024**2:.1f} MiB)',flush=True)
