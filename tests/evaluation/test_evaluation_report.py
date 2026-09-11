"""Synthetic end-to-end raw-score -> bootstrap -> figures/report -> export test."""
from pathlib import Path
import shutil
import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from src.evaluation.common import PES,SEEDS,write_json,sha,evidence,digest,read_json
from src.evaluation.pipeline import jobs,job_path
from src.evaluation.statistics import analyze
from src.evaluation.report import figures,report,export_results
from test_evaluation import protocol

class ReportIntegrationTests(unittest.TestCase):
    def test_synthetic_full_reporting_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            w={'identity':'SYNTHETIC_FIXTURE_NOT_RESEARCH','protocol':protocol(),'runs':[]}
            for pe in PES:
                for seed in SEEDS:
                    w['runs'].append({'run_id':f'{pe}-{seed}','pe_method':pe,'model_seed':seed,
                                     'milestones':{m:{'actual_tokens':n} for m,n in zip(['5m','10m','20m','50m'],[5046272,10027008,20054016,50000000])}})
            ev={}
            for split in ['validation','test']:
                paths=[]
                for run,m,kind,c in jobs(w,split):
                    p=job_path(work,split,run,m,kind,c);p.parent.mkdir(parents=True,exist_ok=True)
                    defined=kind!='length' or run['pe_method']!='learned' or c==512
                    meta={'workspace_identity':w['identity'],'status':'complete' if defined else 'not_applicable',
                          'actual_tokens':run['milestones'][m]['actual_tokens'],'pe':run['pe_method'],'seed':run['model_seed'],
                          'split':split,'kind':kind,'milestone':m,'context':c}
                    if defined:
                        counts=np.array([3,7,19,4,9,11]) if kind=='primary' else np.full(6,256)
                        base=2+{'learned':.04,'sinusoidal':.02,'rope':0.,'alibi':.06,'nope':.08}[run['pe_method']]+SEEDS.index(run['model_seed'])*.002
                        base+=.5*(1-run['milestones'][m]['actual_tokens']/50_000_000)
                        if kind=='length':base+=.1-(c/2048)*.03
                        losses=counts*(base+np.array([0,.1,-.1,.2,-.05,.12]))
                        np.savez_compressed(p.with_suffix('.npz'),document_id=np.array([f'{i:064x}' for i in range(6)],dtype='S64'),
                                            token_count=counts,nll_sum=losses)
                        meta.update(nll=float(losses.sum()/counts.sum()),ppl=float(np.exp(losses.sum()/counts.sum())),
                                    document_count=6,token_count=int(counts.sum()),raw_sha256=sha(p.with_suffix('.npz')))
                        paths.append(p.with_suffix('.npz'))
                    write_json(p,meta);paths.append(p)
                ev[split]=evidence(work,paths)
            write_json(work/'workspace.json',w)
            write_json(work/'runtime.json',{'fixture':True})
            write_json(work/'smoke.json',{'status':'pass','fixture':True})
            write_json(work/'validation_complete.json',{'evidence':ev['validation']})
            write_json(work/'validation_lock.json',{'workspace_identity':w['identity'],'validation_evidence':ev['validation']})
            write_json(work/'final_test_state.json',{'status':'completed','fixture':True})
            write_json(work/'final_test_manifest.json',{'status':'complete','workspace_identity':w['identity'],
                       'lock_sha256':sha(work/'validation_lock.json'),'test_evidence':ev['test']})
            with patch('src.evaluation.statistics.load_workspace',return_value=w),patch('src.evaluation.report.load_workspace',return_value=w):
                result=analyze(work)
                self.assertEqual(result['lowest_observed_mean_nll'],'rope')
                self.assertEqual(result['supported_winner_against_all_four'],'rope')
                self.assertEqual((result['best_vs_runner']['a'],result['best_vs_runner']['b']),('rope','sinusoidal'))
                self.assertLess(result['best_vs_runner']['ci_high'],0)
                self.assertEqual(result['ablation_rope_nope']['a'],'rope')
                figures(work);report(work);export_results(work)
                with np.load(work/'tables/bootstrap_replicates.npz') as z:self.assertEqual(z['mean_nll'].shape,(10000,5))
                with zipfile.ZipFile(work/'EVALUATION_REPORT.zip') as z:
                    self.assertFalse(any(n.startswith('EVALUATION_WORK/scores/') and n.endswith('.npz') for n in z.namelist()))
                    self.assertIn('EVALUATION_WORK/report/Results_Analysis.md',z.namelist())
                with zipfile.ZipFile(work/'EVALUATION_RESULTS.zip') as z:
                    self.assertEqual(len([n for n in z.namelist() if '/scores/test/' in n and n.endswith('.npz')]),90)
                    self.assertFalse(any('tokens.bin' in n for n in z.namelist()))
                    extracted=work/'offline';z.extractall(extracted)
                # Full export can reproduce statistics without validation raw arrays or source datasets.
                offline=extracted/'EVALUATION_WORK'
                rerun=analyze(offline)
                self.assertEqual(rerun['primary'],result['primary'])
            if os.environ.get('EVALUATION_SYNTHETIC_QA_DIR'):
                dest=Path(os.environ['EVALUATION_SYNTHETIC_QA_DIR']);dest.mkdir(parents=True,exist_ok=True)
                for p in (work/'figures').glob('*.png'):shutil.copy2(p,dest/p.name)
                (dest/'SYNTHETIC_ONLY.txt').write_text('Synthetic implementation QA only. Not research results.\n')

if __name__=='__main__':unittest.main()
