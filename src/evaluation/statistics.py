"""Token-weighted paired document bootstrap, conditional on the five trained seeds."""
from __future__ import annotations
import csv
import itertools
import math
from pathlib import Path
import time
import numpy as np
from .common import PES,SEEDS,MILESTONES,require,read_json,write_json,sha,atomic_file,verify_evidence
from .inputs import load_workspace
from .pipeline import job_path

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p,kind='stable')
    adjusted=np.minimum(1.,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    out=np.empty_like(p);out[order]=adjusted
    return out

def paired_bootstrap(sums,counts,resamples=10000,seed=5,batch_size=4,progress=True):
    """sums[method,seed,document], common counts[document]. Same draws for every arm.

    Average seed sums before resampling: algebraically identical to resampling
    each seed with the same document indices because denominators are shared.
    """
    sums=np.asarray(sums,dtype=np.float64);counts=np.asarray(counts,dtype=np.float64)
    require(sums.ndim==3 and sums.shape[2]==len(counts) and np.all(counts>0),'Invalid paired bootstrap input')
    require(np.isfinite(sums).all() and np.isfinite(counts).all(),'Nonfinite bootstrap input')
    means=sums.mean(axis=1);n=len(counts);rng=np.random.default_rng(seed)
    out=np.empty((resamples,sums.shape[0]),dtype=np.float64);last=time.monotonic()
    for begin in range(0,resamples,batch_size):
        end=min(resamples,begin+batch_size)
        idx=rng.integers(0,n,size=(end-begin,n))
        den=counts[idx].sum(axis=1)
        # bounded memory; rows share precisely the same sampled document indices
        for method in range(sums.shape[0]):out[begin:end,method]=means[method][idx].sum(axis=1)/den
        if progress and time.monotonic()-last>20:
            print(f'Paired document bootstrap: {end:,}/{resamples:,}',flush=True);last=time.monotonic()
    return out

def compare(seed_nll,draws,min_practical_delta_nll=0.01):
    means=seed_nll.mean(axis=1);rows=[]
    for i,j in itertools.combinations(range(len(PES)),2):
        observed=float(means[i]-means[j]);delta=draws[:,i]-draws[:,j]
        lo,hi=np.quantile(delta,[.025,.975])
        # Approximate null-centered two-sided paired document bootstrap test.
        p=float((1+np.count_nonzero(np.abs(delta-observed)>=abs(observed)))/(len(delta)+1))
        direction=np.sign(observed)
        consistent=int(np.count_nonzero(np.sign(seed_nll[i]-seed_nll[j])==direction)) if direction else 0
        rows.append({'a':PES[i],'b':PES[j],'delta_nll_a_minus_b':observed,'ci_low':float(lo),'ci_high':float(hi),
                     'relative_ppl_change_a_vs_b_percent':100*math.expm1(observed),
                     'p_uncorrected':p,'consistent_seeds':consistent,
                     'min_practical_delta_nll':min_practical_delta_nll,
                     'passes_practical_effect':bool(abs(observed)>=min_practical_delta_nll)})
    for r,p in zip(rows,holm([r['p_uncorrected'] for r in rows])):
        r['p_holm']=float(p)
        r['statistically_supported_direction']=bool(p<.05 and (r['ci_high']<0 or r['ci_low']>0) and r['consistent_seeds']>=4)
        r['supported_direction']=r['statistically_supported_direction'] and r['passes_practical_effect']
    return rows

def orient_pair(rows,a,b):
    """Canonical A-B contrast; reverse CI endpoints as well as the point estimate."""
    found=[r for r in rows if (r['a'],r['b']) in [(a,b),(b,a)]]
    require(len(found)==1,f'Expected one unique pair for {a}/{b}')
    r=dict(found[0])
    if r['a']!=a:
        r.update(a=a,b=b,delta_nll_a_minus_b=-r['delta_nll_a_minus_b'],
                 ci_low=-found[0]['ci_high'],ci_high=-found[0]['ci_low'])
    r['relative_ppl_change_a_vs_b_percent']=100*math.expm1(r['delta_nll_a_minus_b'])
    return r

def ranked_rows(values):
    """Average ranks for ties, without an additional scientific dependency."""
    values=np.asarray(values)
    return np.array([1+np.count_nonzero(values<v)+(np.count_nonzero(values==v)-1)/2 for v in values],dtype=float)

def learning_summary(learning):
    rows=[];by_milestone={}
    initial={pe:next(r['mean_nll'] for r in learning if r['pe']==pe and r['milestone']=='5m') for pe in PES}
    for milestone in MILESTONES:
        group=[next(r for r in learning if r['pe']==pe and r['milestone']==milestone) for pe in PES]
        ranks=ranked_rows([r['mean_nll'] for r in group])
        minimum=min(r['mean_nll'] for r in group)
        by_milestone[milestone]=[r['pe'] for r in group if r['mean_nll']==minimum]
        for r,rank in zip(group,ranks):
            rows.append({**r,'mean_ppl_from_nll':math.exp(r['mean_nll']),'descriptive_rank':float(rank),
                         'nll_reduction_from_5m':initial[r['pe']]-r['mean_nll']})
    return rows,{'lowest_observed_methods_by_milestone':by_milestone,
                 'leading_method_changes_with_budget':len({tuple(v) for v in by_milestone.values()})>1,
                 'interpretation':'Descriptive points from the same training trajectories; not separate from-scratch runs or interpolated threshold-crossing estimates.',
                 'target_nll_threshold':None,'threshold_crossing_status':'No absolute target NLL was fixed in the supplied plans; no target or crossing time is invented.'}

def write_csv(path,rows):
    require(bool(rows),'Cannot write empty result table')
    with atomic_file(path,'w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def matrix(work,w,split='test',kind='primary',context=512):
    ids=None;counts=None;sums=np.empty((len(PES),len(SEEDS)),dtype=object)
    for run in w['runs']:
        path=job_path(work,split,run,'50m',kind,context)
        meta=read_json(path)
        require(meta['status']=='complete','Expected defined primary score')
        require(sha(path.with_suffix('.npz'))==meta['raw_sha256'],'Raw score hash mismatch')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            current_ids=z['document_id'];current_counts=z['token_count'];current_sums=z['nll_sum']
            if ids is None:ids=current_ids;counts=current_counts
            require(np.array_equal(ids,current_ids) and np.array_equal(counts,current_counts),'Document pairing/count mismatch')
            require(current_sums.shape==counts.shape and np.isfinite(current_sums).all() and (counts>0).all(),'Invalid document losses')
            require(np.isclose(current_sums.sum()/counts.sum(),meta['nll'],rtol=1e-12,atol=1e-12),'Raw/summary NLL mismatch')
            sums[PES.index(run['pe_method']),SEEDS.index(run['model_seed'])]=current_sums
    require(len(np.unique(ids))==len(ids),'Duplicate document IDs')
    return np.stack([np.stack(row.tolist()) for row in sums]),counts,ids

def final_evidence(work,w):
    final=read_json(Path(work)/'final_test_manifest.json');lock=read_json(Path(work)/'validation_lock.json')
    require(final['status']=='complete' and final['workspace_identity']==w['identity'] and
            final['lock_sha256']==sha(Path(work)/'validation_lock.json'),'Final evidence identity mismatch')
    verify_evidence(work,final['test_evidence'])
    # Full export keeps all validation summaries; primary bootstrap uses raw test arrays.
    verify_evidence(work,{n:h for n,h in lock['validation_evidence'].items() if n.endswith('.json')})
    return final

def analyze(work):
    work=Path(work);w=load_workspace(work);final_evidence(work,w)
    sums,counts,ids=matrix(work,w)
    seed_nll=sums.sum(axis=2)/counts.sum()
    p=w['protocol'];draws=paired_bootstrap(sums,counts,p['bootstrap_resamples'],p['bootstrap_seed'],p['bootstrap_batch_size'])
    pairs=compare(seed_nll,draws,p['min_practical_delta_nll']);primary=[]
    for i,pe in enumerate(PES):
        lo,hi=np.quantile(draws[:,i],[.025,.975]);mean=float(seed_nll[i].mean())
        primary.append({'pe':pe,'mean_nll':mean,'seed_sd_nll':float(seed_nll[i].std(ddof=1)),
                        'ppl_exp_mean_nll':math.exp(mean),'mean_seed_ppl':float(np.exp(seed_nll[i]).mean()),
                        'conditional_ci_low':float(lo),'conditional_ci_high':float(hi),
                        'documents':len(ids),'targets_per_seed':int(counts.sum()),
                        **{f'nll_seed_{s}':float(seed_nll[i,j]) for j,s in enumerate(SEEDS)}})
    learning=[];length=[]
    for pe in PES:
        runs=[r for r in w['runs'] if r['pe_method']==pe]
        for m in MILESTONES:
            rows=[read_json(job_path(work,'validation',r,m,'primary',512)) for r in runs]
            values=np.array([r['nll'] for r in rows])
            require(len({r['actual_tokens'] for r in rows})==1,'Training milestone mismatch across seeds')
            learning.append({'pe':pe,'milestone':m,'actual_tokens':rows[0]['actual_tokens'],
                             'mean_nll':float(values.mean()),'seed_sd_nll':float(values.std(ddof=1))})
        for split in ['validation','test']:
            for c in p['length_contexts']:
                rows=[read_json(job_path(work,split,r,'50m','length',c)) for r in runs]
                defined=pe!='learned' or c==512
                require(all(r['status']==('complete' if defined else 'not_applicable') for r in rows),'Length status mismatch')
                if defined:
                    values=np.array([r['nll'] for r in rows])
                    require(len({r['document_count'] for r in rows})==1 and all(r['token_count']==256*r['document_count'] for r in rows),'Length target counts mismatch')
                length.append({'split':split,'pe':pe,'context':c,'status':'complete' if defined else 'not_applicable',
                               'mean_nll':float(values.mean()) if defined else None,
                               'seed_sd_nll':float(values.std(ddof=1)) if defined else None,
                               'documents':rows[0]['document_count'] if defined else None})
    ordered=sorted(primary,key=lambda r:r['mean_nll'])
    best=ordered[0]['pe'];runner=ordered[1]['pe']
    related=[orient_pair(pairs,best,r['pe']) for r in ordered[1:]]
    for r in related:
        other=next(v for v in primary if v['pe']==r['b'])
        require(np.isclose(r['delta_nll_a_minus_b'],ordered[0]['mean_nll']-other['mean_nll'],rtol=1e-12,atol=1e-12),'Canonical contrast/primary mean mismatch')
    best_runner=next(r for r in related if r['b']==runner)
    ablation=orient_pair(pairs,'rope','nope')
    winner=best if all(r['supported_direction'] for r in related) else None
    sample_efficiency,trajectory=learning_summary(learning)
    seed_ranks=[]
    for j,seed in enumerate(SEEDS):
        ranks=ranked_rows(seed_nll[:,j]);minimum=float(seed_nll[:,j].min())
        for i,pe in enumerate(PES):seed_ranks.append({'seed':seed,'pe':pe,'nll':float(seed_nll[i,j]),
                                                     'rank':float(ranks[i]),'tied_for_lowest':bool(seed_nll[i,j]==minimum)})
    result={'primary':primary,'pairwise':pairs,'learning':learning,'length':length,
            'lowest_observed_mean_nll':best,'supported_winner_against_all_four':winner,
            'best_vs_runner':best_runner,'best_vs_others':related,'ablation_rope_nope':ablation,
            'min_practical_delta_nll':p['min_practical_delta_nll'],
            'sample_efficiency':sample_efficiency,'trajectory_summary':trajectory,'seed_ranks':seed_ranks,
            'bootstrap_resamples':p['bootstrap_resamples'],'bootstrap_seed':p['bootstrap_seed'],
            'uncertainty_scope':'Paired document percentile intervals conditional on five fixed trained seeds; seed SD separately reports initialization variation. No seed-population CI is claimed.',
            'p_value_method':'Approximate two-sided null-centered paired document bootstrap, plus-one correction; Holm family of all 10 comparisons.',
            'length_scope':'Documents >=2048 tokens including EOD; identical indices 1792..2047 of the first 2048 tokens are scored at all contexts. This differs from the all-document primary estimand.',
            'final_manifest_sha256':sha(work/'final_test_manifest.json')}
    tables=work/'tables';tables.mkdir(exist_ok=True)
    for name,rows in [('primary',primary),('pairwise',pairs),('learning',learning),('length',length),
                      ('best_vs_others',related),('ablation_rope_nope',[ablation]),
                      ('sample_efficiency',sample_efficiency),('seed_ranks',seed_ranks)]:write_csv(tables/(name+'.csv'),rows)
    with atomic_file(tables/'bootstrap_replicates.npz') as f:np.savez_compressed(f,method=np.array(PES),mean_nll=draws)
    write_json(tables/'analysis.json',result)
    print(f'Analysis complete. Lowest observed NLL: {best}; supported winner against all four: {winner}',flush=True)
    return result
