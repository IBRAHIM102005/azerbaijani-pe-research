"""Synthetic oracles: no real test text, trained checkpoints or headline claims."""
import io
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.evaluation.common import ROOT,PES,SEEDS,require,digest,sha,read_json,write_json,evidence
from src.evaluation.data import primary_windows,length_window
from src.evaluation.metrics import score_documents,load_model
from src.evaluation.inputs import tar_members,load_workspace
from src.evaluation.pipeline import evaluate_split,final_test,job_path,jobs
from src.evaluation.statistics import holm,paired_bootstrap,compare,matrix,orient_pair,learning_summary
from src.models.config import ModelConfig
from src.models.transformer import build_model

torch.set_num_threads(1)

def protocol():
    p=read_json(ROOT/'configs/evaluation_protocol.json');p['precision']='fp32';return p

class Cache:
    def __init__(self,documents):
        self.documents=[np.asarray(x,dtype=np.uint16) for x in documents]
        self.ids=np.array([f'{i:064x}' for i in range(len(documents))],dtype='S64')
        self.lengths=np.array([len(x) for x in documents])
        self.meta={'fixture':True,'identity':digest([x.tolist() for x in self.documents])}
    def __len__(self):return len(self.documents)
    def __getitem__(self,i):return self.documents[i]

class TransitionModel(torch.nn.Module):
    """Nonuniform exact oracle: next logits depend on current token, catches shifts."""
    def __init__(self):
        super().__init__()
        self.register_buffer('logits',torch.tensor([[1.,0.,-1.,2.],[0.,2.,-1.,1.],[-2.,1.,3.,0.],[2.,-1.,0.,1.]]))
        self.calls=0
    def forward(self,x):
        self.calls+=1
        return self.logits[x],None

def oracle(tokens,model,first=1):
    logp=torch.log_softmax(model.logits.double(),dim=-1).numpy()
    return sum(-logp[int(tokens[i-1]),int(tokens[i])] for i in range(first,len(tokens)))

class MetricTests(unittest.TestCase):
    def test_primary_exact_target_coverage(self):
        for context,stride in [(8,4),(512,256)]:
            for n in [2,3,7,8,9,511,512,513,767,768,769,1025,4099]:
                scored=[]
                for start,end,first in primary_windows(n,context,stride):
                    self.assertLessEqual(end-start,context);self.assertGreaterEqual(first,1)
                    scored.extend(range(start+first,end))
                self.assertEqual(scored,list(range(1,n)))

    def test_matched_context_targets(self):
        for c in [512,1024,2048]:
            start,end,first=length_window(9000,c)
            self.assertEqual(list(range(start+first,end)),list(range(1792,2048)))

    def test_nonuniform_nll_padding_shift_and_weighting(self):
        docs=[[0,1,2,1],([0,3,1,2]*270)+[1],[3,1]]
        cache=Cache(docs);model=TransitionModel();p=protocol()
        raw,meta=score_documents(model,cache,np.arange(3),'primary',512,p,'cpu')
        expected=np.array([oracle(x,model) for x in docs])
        np.testing.assert_allclose(raw['nll_sum'],expected,rtol=1e-7,atol=2e-5)
        np.testing.assert_array_equal(raw['token_count'],np.array([len(x)-1 for x in docs]))
        self.assertAlmostEqual(meta['nll'],expected.sum()/sum(len(x)-1 for x in docs),places=6)
        self.assertNotAlmostEqual(meta['nll'],np.mean(expected/np.array([len(x)-1 for x in docs])),places=3)

    def test_length_score_oracle(self):
        docs=[([0,3,1,2]*600)+[1],([2,1,3,0]*520)+[1]];cache=Cache(docs);model=TransitionModel()
        expected=np.array([oracle(x[:2048],model,1792) for x in docs])
        for c in [512,1024,2048]:
            raw,_=score_documents(model,cache,np.arange(2),'length',c,protocol(),'cpu')
            np.testing.assert_array_equal(raw['token_count'],[256,256])
            np.testing.assert_allclose(raw['nll_sum'],expected,rtol=1e-7)

    def test_actual_model_causality_all_five_methods(self):
        for pe in PES:
            model=build_model(ModelConfig(pe_type=pe,init_seed=17,vocab_size=32,n_layer=1,n_head=2,d_model=16,d_ff=32,max_seq_len=16)).eval()
            x=torch.arange(12).reshape(1,-1);y=x.clone();y[:,7:]=21
            with torch.no_grad():a,_=model(x);b,_=model(y)
            torch.testing.assert_close(a[:,:7],b[:,:7],rtol=0,atol=0)

class StatisticsTests(unittest.TestCase):
    def test_canonical_forward_and_reverse_pair_have_identical_meaning(self):
        forward={'a':'rope','b':'sinusoidal','delta_nll_a_minus_b':-.3,'ci_low':-.36,'ci_high':-.25,
                 'consistent_seeds':5,'p_holm':.001,'passes_practical_effect':True,'supported_direction':True}
        reverse={**forward,'a':'sinusoidal','b':'rope','delta_nll_a_minus_b':.3,'ci_low':.25,'ci_high':.36}
        a=orient_pair([forward],'rope','sinusoidal');b=orient_pair([reverse],'rope','sinusoidal')
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['relative_ppl_change_a_vs_b_percent'],100*math.expm1(-.3))

    def test_small_statistically_supported_effect_fails_practical_rule(self):
        seed_nll=np.array([[2.]*5,[2.005]*5,[2.05]*5,[2.1]*5,[2.2]*5])
        draws=np.tile(seed_nll.mean(axis=1),(10000,1))
        pair=orient_pair(compare(seed_nll,draws),'learned','sinusoidal')
        self.assertTrue(pair['statistically_supported_direction'])
        self.assertFalse(pair['passes_practical_effect']);self.assertFalse(pair['supported_direction'])

    def test_learning_summary_retains_ties_and_does_not_invent_threshold(self):
        rows=[{'pe':pe,'milestone':m,'actual_tokens':n,'mean_nll':3.-j*.2+(0 if pe in ['rope','nope'] else .1),
               'seed_sd_nll':.01} for j,(m,n) in enumerate(zip(['5m','10m','20m','50m'],[5e6,10e6,20e6,50e6])) for pe in PES]
        table,summary=learning_summary(rows)
        self.assertEqual(len(table),20)
        self.assertEqual(summary['lowest_observed_methods_by_milestone']['50m'],['rope','nope'])
        self.assertIsNone(summary['target_nll_threshold'])
        self.assertAlmostEqual(next(r['nll_reduction_from_5m'] for r in table if r['pe']=='rope' and r['milestone']=='50m'),.6)

    def test_token_weighted_shared_bootstrap_and_batch_invariance(self):
        sums=np.array([[[2.,30.],[4.,32.]],[[4.,40.],[6.,42.]]]);counts=np.array([1,10])
        a=paired_bootstrap(sums,counts,100,5,1,False);b=paired_bootstrap(sums,counts,100,5,7,False)
        np.testing.assert_array_equal(a,b)
        possible=np.array([[3.,5.],[31/10,41/10],[(3+31)/11,(5+41)/11]])
        for row in a:self.assertTrue(any(np.allclose(row,p) for p in possible))
        self.assertTrue(np.all(a[:,0]<a[:,1]))

    def test_holm_manual_example(self):
        np.testing.assert_allclose(holm([.03,.001,.04,.2]),[.09,.004,.09,.2])

    def test_identical_methods_never_winner(self):
        seed_nll=np.ones((5,5))*2;draws=np.ones((100,5))*2
        for r in compare(seed_nll,draws):
            self.assertEqual(r['p_holm'],1);self.assertEqual(r['ci_low'],0);self.assertFalse(r['supported_direction'])

    def test_pairing_rejects_reordered_documents_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);w={'runs':[]}
            for pe in PES:
                for seed in SEEDS:
                    run={'run_id':f'{pe}-{seed}','pe_method':pe,'model_seed':seed};w['runs'].append(run)
                    p=job_path(work,'test',run,'50m','primary',512);p.parent.mkdir(parents=True)
                    np.savez_compressed(p.with_suffix('.npz'),document_id=np.array(['a','b']),token_count=np.array([1,10]),nll_sum=np.array([2.,30.]))
                    write_json(p,{'status':'complete','nll':32/11,'raw_sha256':sha(p.with_suffix('.npz'))})
            arr,counts,ids=matrix(work,w);self.assertEqual(arr.shape,(5,5,2))
            np.savez_compressed(p.with_suffix('.npz'),document_id=np.array(['b','a']),token_count=np.array([1,10]),nll_sum=np.array([2.,30.]))
            meta=read_json(p);meta['raw_sha256']=sha(p.with_suffix('.npz'));write_json(p,meta)
            with self.assertRaisesRegex(RuntimeError,'pairing'):matrix(work,w)

class CheckpointAndGateTests(unittest.TestCase):
    def test_real_m3_writer_and_tar_loader_all_five_methods(self):
        from scripts.m3_train import save_model_only_checkpoint
        with tempfile.TemporaryDirectory() as d:
            for pe in PES:
                config=ModelConfig(pe_type=pe,init_seed=17,vocab_size=32,n_layer=1,n_head=2,d_model=16,d_ff=32,max_seq_len=16)
                model=build_model(config).eval()
                with torch.no_grad():model.wte.weight.add_(.17)
                path=Path(d)/(pe+'.pt')
                save_model_only_checkpoint(path,model=model,run_id=pe,pe_type=pe,init_seed=17,nominal_tokens=50_000_000,actual_tokens=50_000_000)
                archive=Path(d)/(pe+'.tar');member=pe+'/50m_model.pt'
                with tarfile.open(archive,'w') as t:t.add(path,arcname=member)
                tar_members(archive,{member:sha(path)})
                run={'run_id':pe,'pe_method':pe,'model_seed':17,'config':config.to_dict(),
                     'milestones':{'50m':{'nominal_tokens':50_000_000,'actual_tokens':50_000_000}},
                     'checkpoints':{'50m':{'path':str(archive),'member':member,'sha256':sha(path)}}}
                restored=load_model(run,'50m','cpu');x=torch.arange(10).reshape(1,-1)
                with torch.no_grad():a,_=model(x);b,_=restored(x)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                run['model_seed']=42
                with self.assertRaisesRegex(RuntimeError,'payload'):load_model(run,'50m','cpu')

    def test_tar_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name,symlink in [('../unsafe',False),('run/50m_model.pt',True)]:
                p=Path(d)/'bad.tar';info=tarfile.TarInfo(name)
                if symlink:info.type=tarfile.SYMTYPE;info.linkname='/outside'
                else:info.size=1
                with tarfile.open(p,'w') as t:t.addfile(info,None if symlink else io.BytesIO(b'x'))
                with self.assertRaisesRegex(RuntimeError,'Unsafe'):tar_members(p,{name:'irrelevant'})

    def test_committed_jobs_resume_without_model_loading_and_corruption_stops(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);cache=Cache([[0,1,2,1],[1,3,1]])
            run={'run_id':'toy','pe_method':'nope','model_seed':17,'sha_50m':'toy',
                 'milestones':{'50m':{'actual_tokens':50_000_000}}}
            w={'identity':'toy','protocol':protocol(),'runs':[run]};rt={'toy':True}
            one=[(run,'50m','primary',512)]
            with patch('src.evaluation.pipeline.jobs',return_value=one),patch('src.evaluation.pipeline.load_model',return_value=TransitionModel()) as loader:
                evaluate_split(work,w,cache,'test','cpu',rt);self.assertEqual(loader.call_count,1)
                loader.reset_mock();evaluate_split(work,w,cache,'test','cpu',rt);loader.assert_not_called()
                path=job_path(work,'test',run,'50m','primary',512).with_suffix('.npz');path.write_bytes(b'corrupted')
                with self.assertRaisesRegex(RuntimeError,'SHA-256'):evaluate_split(work,w,cache,'test','cpu',rt)
                loader.assert_not_called()

    def test_final_test_gate_and_completed_no_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);w={'identity':'toy'}
            with patch('src.evaluation.pipeline.load_workspace',return_value=w),patch('src.evaluation.pipeline.build_cache') as builder:
                with self.assertRaisesRegex(RuntimeError,'Run lock'):final_test(work,'cpu')
                builder.assert_not_called()
                write_json(work/'validation_lock.json',{'workspace_identity':'toy','validation_evidence':{}})
                write_json(work/'final_test_manifest.json',{'status':'complete','workspace_identity':'toy',
                           'lock_sha256':sha(work/'validation_lock.json'),'test_evidence':{}})
                final_test(work,'cpu');builder.assert_not_called()

    def test_workspace_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w={'protocol':{'evaluation_seed':5}};w['identity']=digest(w);w['protocol']['evaluation_seed']=42
            write_json(Path(d)/'workspace.json',w)
            with self.assertRaisesRegex(RuntimeError,'identity'):load_workspace(d,require_weights=False,verify_source=False)

if __name__=='__main__':unittest.main()
