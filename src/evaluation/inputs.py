"""Resolve the exact frozen M1/M3 files by content, including numbered uploads."""
from __future__ import annotations
import csv
import hashlib
import io
from pathlib import Path
import tarfile
from .common import (ROOT,HANDOFF,PES,SEEDS,MILESTONES,require,read_json,write_json,
                     sha,digest,check_file,protocol,source_identity,now)

def find_file(spec, roots, suffix):
    candidates = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        direct = root/spec['path']
        if direct.is_file(): candidates.append(direct)
        if root.is_file(): candidates.append(root)
        elif root.is_dir(): candidates.extend(root.rglob('*'+suffix))
    for p in dict.fromkeys(candidates):
        if (spec.get('bytes') is None or p.stat().st_size==spec['bytes']) and sha(p)==spec['sha256']:
            return {**spec, 'path':str(p)}
    raise RuntimeError(f"Cannot find exact {spec['path']}; give its parent using --input-dir/--checkpoint-dir.")

def tar_members(path, expected):
    with tarfile.open(path, 'r:') as t:
        members = t.getmembers()
        require(len(members)==len(expected), f'Unexpected TAR member count: {path}')
        require({m.name for m in members}==set(expected), f'Unexpected/duplicate TAR names: {path}')
        for m in members:
            require(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,
                    f'Unsafe TAR member: {m.name}')
            with t.extractfile(m) as f:
                require(hashlib.file_digest(f,'sha256').hexdigest()==expected[m.name], f'Bad member SHA: {m.name}')

def checkpoint_bytes(ref):
    if ref.get('member'):
        with tarfile.open(ref['path'], 'r:') as t:
            m = t.getmember(ref['member'])
            require(m.isfile() and 0<m.size<200_000_000, 'Invalid model-only checkpoint member')
            with t.extractfile(m) as f: data = f.read()
    else:
        data = Path(ref['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest()==ref['sha256'], f"Checkpoint SHA mismatch: {ref['path']}")
    return io.BytesIO(data)

def prepare(work, input_dirs, checkpoint_dirs, metadata_only=False):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    sums = {line.split(maxsplit=1)[1].strip():line.split()[0]
            for line in (HANDOFF/'SHA256SUMS.txt').read_text().splitlines() if line.strip()}
    for name in ['m3_handoff_manifest.json','m3_handoff_runs.csv','training_data_contract.json']:
        check_file(HANDOFF/name, sums[name])
    origin = read_json(HANDOFF/'m4_source_origin.json')
    for name, expected in origin['verified_source_hashes'].items():
        check_file(ROOT/name, expected)
    manifest = read_json(HANDOFF/'m3_handoff_manifest.json')
    plan = read_json(ROOT/'results/manifests/m3_run_plan.json')
    contract = read_json(HANDOFF/'training_data_contract.json')['artifacts']
    rows = list(csv.DictReader((HANDOFF/'m3_handoff_runs.csv').open()))
    require(manifest['status']=='complete' and len(manifest['runs'])==25 and len(rows)==25, 'M3 matrix incomplete')
    require({(r['pe_method'],r['model_seed']) for r in manifest['runs']}=={(p,s) for p in PES for s in SEEDS}, 'Unexpected matrix')
    byplan = {r['run_id']:r for r in plan['runs']}
    bycsv = {r['run_id']:r for r in rows}
    runs = []
    for r in manifest['runs']:
        rid = r['run_id']; pr = byplan[rid]
        require(all(str(v)==bycsv[rid][k] for k,v in r.items()), f'CSV/JSON mismatch: {rid}')
        require(r['tokens_seen']==50_000_000 and r['optimizer_steps']==763 and r['data_seed']==2026,
                f'Incomplete training: {rid}')
        config = read_json(ROOT/f"configs/pe/{r['pe_method']}.json")
        config['init_seed'] = r['model_seed']
        meta = read_json(ROOT/'results/runs'/rid/'metadata.json')
        done = read_json(ROOT/'results/runs'/rid/'completed.json')
        require(digest(config)==pr['config_sha256']==meta['resolved_config_hash'], f'Config mismatch: {rid}')
        require(done['status']=='completed' and done['tokens_seen']==50_000_000 and
                done['optimizer_steps']==763 and meta['exit_code']==0, f'Completion mismatch: {rid}')
        require(meta['git_commit']==manifest['source_commit']==origin['training_commit'], 'Source commit mismatch')
        for m in MILESTONES:
            require(r['sha_'+m]==meta['checkpoint_hashes'][m+'_model']==done['checkpoint_hashes'][m+'_model'], 'Checkpoint metadata mismatch')
        runs.append({**r,'config':config,'milestones':{c['label']:c for c in pr['checkpoints']},'checkpoints':{}})
    data = {}
    for split in ['validation','test']:
        data[split] = {}
        for key in ['manifests','processed_corpus']:
            data[split][key] = find_file(contract[key][split], [*input_dirs,ROOT], '.parquet')
            print(f"Verified {split} {key}", flush=True)
    tokenizer = find_file(contract['tokenizer']['tokenizer.model'],[ROOT,*input_dirs], '.model')
    if not metadata_only:
        for m in MILESTONES:
            name = f'M3_CHECKPOINTS_{m.upper()}.tar'
            # TARs are verified in full before any model is loaded; no extraction required.
            tar = None
            for d in checkpoint_dirs:
                base = Path(d).expanduser().resolve()
                candidates = [base] if base.is_file() else sorted(base.rglob('*.tar')) if base.is_dir() else []
                matching = [p for p in candidates if p.name==name or p.name.startswith(name[:-4]+'(')]
                if matching:
                    tar = matching[0]; check_file(tar,sums[name]); break
            if tar:
                expected = {f"{r['run_id']}/{m}_model.pt":r['sha_'+m] for r in runs}
                tar_members(tar,expected)
                for r in runs:
                    r['checkpoints'][m]={'path':str(tar),'member':f"{r['run_id']}/{m}_model.pt",'sha256':r['sha_'+m]}
            else:
                for r in runs:
                    candidates = [Path(r['checkpoint_'+m])]
                    for d in checkpoint_dirs:
                        b=Path(d).expanduser().resolve()
                        candidates += [b/r['run_id']/f'{m}_model.pt', b/r['run_id']/'checkpoints/milestones'/f'{m}_model.pt',
                                       b/'results/runs'/r['run_id']/'checkpoints/milestones'/f'{m}_model.pt']
                    found = next((p for p in candidates if p.is_file()),None)
                    require(found is not None, f'Missing {name} or loose {r["run_id"]}/{m}_model.pt')
                    check_file(found,r['sha_'+m])
                    r['checkpoints'][m]={'path':str(found.resolve()),'sha256':r['sha_'+m]}
            print(f'Verified all 25 {m} checkpoint files', flush=True)
    w={'schema_version':1,'source':source_identity(),'protocol':protocol(),'runs':runs,
       'data':data,'tokenizer':tokenizer,'source_origin':origin,
       'checkpoints_verified':not metadata_only,'handoff_sha256':sha(HANDOFF/'m3_handoff_manifest.json')}
    w['identity']=digest(w)
    target=work/'workspace.json'
    if target.exists() and read_json(target)['identity']!=w['identity']:
        require(not list((work/'scores').rglob('*.json')) and not (work/'validation_lock.json').exists(),
                'Prepared inputs/protocol changed after scoring. Keep the existing experiment intact.')
    write_json(target,w)
    print(f'Prepared {len(runs)} runs. Checkpoints verified: {not metadata_only}',flush=True)
    return w

def load_workspace(work, require_weights=True, verify_source=True):
    w=read_json(Path(work)/'workspace.json')
    require(digest({k:v for k,v in w.items() if k!='identity'})==w['identity'], 'workspace.json identity mismatch')
    if verify_source:
        require(w['source']==source_identity() and w['protocol']==protocol(), 'Code/protocol changed after prepare')
    if require_weights: require(w['checkpoints_verified'], 'Metadata-only preflight cannot evaluate; run prepare with TAR paths')
    return w
