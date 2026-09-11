#!/usr/bin/env python3
"""Evaluation-only Evaluation command line entry point. See docs/evaluation/README.md."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONHASHSEED','5')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.common import file_lock,read_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['doctor','self-test','prepare','cache-validation','smoke','validate','lock','final-test',
                                        'analyze','figures','report','export','status','run-all'])
    parser.add_argument('--work-dir',type=Path,default=ROOT/'EVALUATION_WORK')
    parser.add_argument('--input-dir',type=Path,action='append',default=[])
    parser.add_argument('--checkpoint-dir',type=Path,action='append',default=[])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args();work=args.work_dir.expanduser().resolve()
    if args.metadata_only and args.command!='prepare':parser.error('--metadata-only is for prepare only')
    if args.command=='doctor':
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pyarrow','sentencepiece','matplotlib']}
        import torch
        versions.update(python=sys.version,cuda_available=torch.cuda.is_available(),cuda=torch.version.cuda)
        print(json.dumps(versions,indent=2))
        if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; select the M3 GPU Python environment')
        return
    if args.command=='self-test':
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests/evaluation'),'-v'],cwd=ROOT,check=True)
        return
    work.mkdir(parents=True,exist_ok=True)
    with file_lock(work/'m4-process.lock'):
        from src.evaluation.inputs import prepare,load_workspace
        from src.evaluation.pipeline import smoke,validate,lock_validation,final_test
        actions={
            'prepare':lambda:prepare(work,args.input_dir,args.checkpoint_dir,args.metadata_only),
            'smoke':lambda:smoke(work,args.device),
            'validate':lambda:validate(work,args.device),
            'lock':lambda:lock_validation(work),
            'final-test':lambda:final_test(work,args.device)}
        def analysis():
            from src.evaluation.statistics import analyze
            return analyze(work)
        def presentation(name):
            from src.evaluation import report
            return getattr(report,name)(work)
        actions.update(analyze=analysis,figures=lambda:presentation('figures'),report=lambda:presentation('report'),export=lambda:presentation('export_results'))
        if args.command=='cache-validation':
            from src.evaluation.data import build_cache
            build_cache(work,load_workspace(work,require_weights=False),'validation')
        elif args.command=='status':
            print(json.dumps({n:read_json(work/n).get('status','present') if (work/n).exists() else 'absent'
                              for n in ['workspace.json','smoke.json','validation_complete.json','validation_lock.json','final_test_state.json','final_test_manifest.json']},indent=2))
            print('Committed score summaries:',len(list((work/'scores').rglob('*.json'))))
        elif args.command=='run-all':
            for name in ['prepare','smoke','validate','lock','final-test','analyze','figures','report','export']:
                if name=='smoke' and (work/'smoke.json').exists():
                    w=load_workspace(work)
                    if read_json(work/'smoke.json')['workspace_identity']==w['identity']:continue
                print(f'\nEvaluation STAGE: {name}',flush=True);actions[name]()
        else:actions[args.command]()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\nInterrupted. Re-run the same command/workspace to resume uncommitted jobs.',file=sys.stderr);sys.exit(130)
    except Exception as exc:
        print(f'Evaluation STOPPED: {type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(1)
