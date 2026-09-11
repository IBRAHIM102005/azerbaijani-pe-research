"""Sequential Evaluation jobs, durable evidence, validation freeze and one final test batch."""
from __future__ import annotations
import contextlib
import gc
from pathlib import Path
import tempfile
import numpy as np
from .common import (MILESTONES,require,read_json,write_json,sha,digest,now,event,
                     atomic_file,file_lock,evidence,verify_evidence)
from .inputs import load_workspace
from .data import build_cache
from .metrics import configure_runtime,load_model,score_documents

def jobs(w,split):
    for run in w['runs']:
        for m in MILESTONES if split=='validation' else ['50m']:
            yield run,m,'primary',512
        for c in w['protocol']['length_contexts']:
            yield run,'50m','length',c

def job_path(work,split,run,m,kind,context):
    return Path(work)/'scores'/split/run['run_id']/f'{m}_{kind}_c{context}.json'

def identity(w,runtime,cache,run,m,kind,c):
    return digest({'workspace':w['identity'],'runtime':runtime,'cache':cache.meta,
                   'run_id':run['run_id'],'checkpoint':run['sha_'+m],'kind':kind,'context':c})

def committed(path,expected=None):
    if not path.exists():return False
    meta=read_json(path)
    if expected is not None:require(meta['identity']==expected,f'Job identity changed: {path}')
    require(meta['status'] in ['complete','not_applicable'],f'Incomplete committed job: {path}')
    if meta['status']=='complete':
        from .common import check_file
        check_file(path.with_suffix('.npz'),meta['raw_sha256'])
    return True

def collect(work,w,split):
    paths=[]
    for run,m,kind,c in jobs(w,split):
        path=job_path(work,split,run,m,kind,c)
        require(committed(path),f'Missing job: {path}')
        meta=read_json(path)
        require(meta['workspace_identity']==w['identity'],'Job belongs to another prepared experiment')
        paths.append(path)
        if meta['status']=='complete':paths.append(path.with_suffix('.npz'))
    return evidence(Path(work),paths)

@contextlib.contextmanager
def runtime_session(work,w,device):
    info=configure_runtime(w['protocol'],device)
    path=Path(work)/'runtime.json'
    if path.exists():require(read_json(path)==info,'Runtime/device changed; resume with the same Python environment and GPU')
    else:write_json(path,info)
    key=digest({'gpu':info.get('gpu_uuid')})[:24] if info['device_type']=='cuda' else 'cpu'
    with file_lock(Path(tempfile.gettempdir())/f'm4-device-{key}.lock'):
        yield info

def release(model,device):
    # The caller must also delete its reference before loading another model.
    import torch
    del model
    gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()

def smoke(work,device):
    w=load_workspace(work)
    with runtime_session(work,w,device) as rt:
        cache=build_cache(work,w,'validation');rows=[]
        for run in w['runs']:
            if run['model_seed']!=17:continue
            model=load_model(run,'50m',device)
            try:
                for kind,c in [('primary',512)]+[('length',c) for c in [512,1024,2048] if run['pe_method']!='learned' or c==512]:
                    idx=np.arange(min(32,len(cache))) if kind=='primary' else np.flatnonzero(cache.lengths>=2048)[:8]
                    _,metrics=score_documents(model,cache,idx,kind,c,w['protocol'],device)
                    rows.append({'pe':run['pe_method'],'kind':kind,'context':c,**metrics})
                    print(f"SMOKE {run['pe_method']} {kind} {c}: NLL={metrics['nll']:.6f}",flush=True)
            finally:
                previous,model=model,None;del previous;release(None,device)
        out={'workspace_identity':w['identity'],'runtime':rt,'rows':rows,'status':'pass','time':now()}
        write_json(Path(work)/'smoke.json',out)
        return out

def evaluate_split(work,w,cache,split,device,rt):
    model=None;loaded=None
    try:
        for run,m,kind,c in jobs(w,split):
            path=job_path(work,split,run,m,kind,c)
            key=identity(w,rt,cache,run,m,kind,c)
            if committed(path,key):
                print(f'SKIP committed {split} {run["pe_method"]} s{run["model_seed"]} {m} {kind} c{c}',flush=True)
                continue
            meta={'identity':key,'workspace_identity':w['identity'],'split':split,'run_id':run['run_id'],
                  'pe':run['pe_method'],'seed':run['model_seed'],'milestone':m,'kind':kind,'context':c,
                  'actual_tokens':run['milestones'][m]['actual_tokens'],'checkpoint_sha256':run['sha_'+m]}
            if kind=='length' and run['pe_method']=='learned' and c>512:
                write_json(path,{**meta,'status':'not_applicable','reason':'Learned table trained only to position 511; no new parameters/interpolation introduced.'})
                continue
            if loaded!=(run['run_id'],m):
                previous,model=model,None;del previous;release(None,device)
                model=load_model(run,m,device);loaded=(run['run_id'],m)
            idx=np.arange(len(cache)) if kind=='primary' else np.flatnonzero(cache.lengths>=2048)
            print(f'RUN {split} {run["pe_method"]} s{run["model_seed"]} {m} {kind} c{c}',flush=True)
            arrays,metrics=score_documents(model,cache,idx,kind,c,w['protocol'],device)
            with atomic_file(path.with_suffix('.npz')) as f:np.savez_compressed(f,**arrays)
            write_json(path,{**meta,**metrics,'status':'complete','raw_sha256':sha(path.with_suffix('.npz')),'completed_at':now()})
            event(work,'job_committed',job=str(path.relative_to(work)),identity=key)
            print(f'COMMITTED NLL={metrics["nll"]:.6f}; PPL={metrics["ppl"]:.4f}',flush=True)
    finally:
        del model;release(None,device)

def validate(work,device):
    work=Path(work);w=load_workspace(work)
    if (work/'validation_lock.json').exists():
        lock=read_json(work/'validation_lock.json')
        require(lock['workspace_identity']==w['identity'],'Validation lock mismatch')
        verify_evidence(work,lock['validation_evidence'])
        print('Validation already frozen and verified. No models evaluated.',flush=True)
        return
    require((work/'smoke.json').exists(),'Run smoke before full validation')
    require(read_json(work/'smoke.json')['workspace_identity']==w['identity'],'Smoke belongs to old inputs')
    with runtime_session(work,w,device) as rt:
        cache=build_cache(work,w,'validation')
        evaluate_split(work,w,cache,'validation',device,rt)
    ev=collect(work,w,'validation')
    write_json(work/'validation_complete.json',{'workspace_identity':w['identity'],'evidence':ev,'time':now()})

def lock_validation(work):
    work=Path(work);w=load_workspace(work)
    path=work/'validation_lock.json'
    if path.exists():
        lock=read_json(path);require(lock['workspace_identity']==w['identity'],'Lock mismatch')
        verify_evidence(work,lock['validation_evidence']);return lock
    require((work/'validation_complete.json').exists(),'All validation jobs must finish before locking')
    require(read_json(work/'smoke.json')['status']=='pass','Missing smoke pass')
    ev=collect(work,w,'validation')
    require(ev==read_json(work/'validation_complete.json')['evidence'],'Validation evidence changed')
    # Byte hashes only: test text/token cache is not opened by this operation.
    from .common import check_file
    for split in w['data'].values():
        for spec in split.values():check_file(spec['path'],spec['sha256'])
    lock={'workspace_identity':w['identity'],'runtime':read_json(work/'runtime.json'),
          'protocol':w['protocol'],'validation_evidence':ev,'time':now(),
          'test_policy':'one_frozen_final_batch_resumable_missing_jobs_only'}
    write_json(path,lock);event(work,'validation_locked',lock_sha256=sha(path))
    print('Validation/protocol frozen. Final test may now run.',flush=True)
    return lock

def final_test(work,device):
    work=Path(work);w=load_workspace(work)
    lockpath=work/'validation_lock.json'
    require(lockpath.exists(),'Run lock after completed validation before final-test')
    lock=read_json(lockpath);locksha=sha(lockpath)
    require(lock['workspace_identity']==w['identity'],'Validation lock belongs to another experiment')
    verify_evidence(work,lock['validation_evidence'])
    finalpath=work/'final_test_manifest.json'
    if finalpath.exists():
        final=read_json(finalpath)
        require(final['status']=='complete' and final['workspace_identity']==w['identity'] and final['lock_sha256']==locksha,'Final test identity mismatch')
        verify_evidence(work,final['test_evidence'])
        write_json(work/'final_test_state.json',{'status':'completed','lock_sha256':locksha})
        print('Final test already complete; verified saved results, no test inference repeated.',flush=True)
        return final
    statepath=work/'final_test_state.json'
    if statepath.exists():require(read_json(statepath)['lock_sha256']==locksha,'Cannot resume under another test lock')
    with runtime_session(work,w,device) as rt:
        require(rt==lock['runtime'],'Runtime changed after validation freeze')
        write_json(statepath,{'status':'running','lock_sha256':locksha,'time':now()})
        event(work,'final_test_started_or_resumed',lock_sha256=locksha)
        try:
            cache=build_cache(work,w,'test',allow_test=True)
            evaluate_split(work,w,cache,'test',device,rt)
            ev=collect(work,w,'test')
            final={'status':'complete','workspace_identity':w['identity'],'lock_sha256':locksha,
                   'test_evidence':ev,'completed_at':now()}
            write_json(finalpath,final)
            write_json(statepath,{'status':'completed','lock_sha256':locksha})
            event(work,'final_test_completed',manifest_sha256=sha(finalpath))
        except BaseException:
            write_json(statepath,{'status':'interrupted','lock_sha256':locksha,'time':now()})
            raise
    return final
