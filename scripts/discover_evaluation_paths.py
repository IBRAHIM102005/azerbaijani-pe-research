#!/usr/bin/env python3
"""Read-only directory discovery. Never opens checkpoints or held-out text."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ARCHIVES=['M3_CHECKPOINTS_'+m+'M.tar' for m in ['5','10','20','50']]

def discover(roots):
    result={'package_roots':[],'checkpoint_directories':[],'loose_checkpoint_roots':[],
            'parquet_directories':[],'archives':[],'warnings':[]}
    seen=set()
    for root in roots:
        root=Path(root).expanduser().resolve()
        if not root.is_dir():
            result['warnings'].append(f'Qovluq mövcud deyil: {root}');continue
        def onerror(error):result['warnings'].append(str(error))
        for base,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
            dirs[:]=[d for d in dirs if d not in {'.git','.venv','venv','__pycache__','node_modules','.cache'}]
            p=Path(base)
            if p in seen:
                dirs[:]=[];continue
            seen.add(p)
            if (p/'scripts/evaluate.py').is_file():result['package_roots'].append(str(p))
            found=[n for n in files if n in ARCHIVES or any(n.startswith(a[:-4]+'(') and n.endswith('.tar') for a in ARCHIVES)]
            if found:
                result['checkpoint_directories'].append(str(p))
                result['archives'].extend(str(p/n) for n in sorted(found))
            if '50m_model.pt' in files and p.name=='milestones' and p.parent.name=='checkpoints':
                result['loose_checkpoint_roots'].append(str(p.parent.parent.parent))
            if any(n.endswith('.parquet') and (n.startswith('validation') or n.startswith('test')) for n in files):
                result['parquet_directories'].append(str(p))
    for key in ['package_roots','checkpoint_directories','loose_checkpoint_roots','parquet_directories','archives']:
        result[key]=sorted(set(result[key]))
    return result

def print_result(result):
    labels={'package_roots':'REPO üçün paket qovluğu','checkpoint_directories':'CHECKPOINT_DIRS üçün TAR qovluğu',
            'loose_checkpoint_roots':'CHECKPOINT_DIRS üçün açılmış run qovluğu',
            'parquet_directories':'INPUT_DIRS üçün Parquet qovluğu','archives':'Tapılan TAR faylları'}
    for key,label in labels.items():
        print('\n'+label+':')
        for p in result[key]:print(' ',p)
        if not result[key]:print('  Tapılmadı')
    if result['warnings']:print('\nQeydlər:\n'+'\n'.join(result['warnings']))
    print('\nBu yalnız fayl/yol axtarışıdır. Dəqiq məzmun uyğunluğunu prepare SHA-256 ilə yoxlayacaq.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,action='append')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    result=discover(args.root or [Path.cwd()])
    if args.json:print(json.dumps(result,indent=2,ensure_ascii=False))
    else:print_result(result)

if __name__=='__main__':main()
