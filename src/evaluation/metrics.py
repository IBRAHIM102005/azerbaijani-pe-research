"""Strict M3 inference and token-weighted document NLL. No training path."""
from __future__ import annotations
import contextlib
import importlib.metadata
import math
import os
import platform
import random
import subprocess
import time
import numpy as np
from .common import require
from .inputs import checkpoint_bytes
from .data import primary_windows,length_window

def configure_runtime(p,device):
    import torch
    torch.set_num_threads(p['cpu_threads'])
    random.seed(p['evaluation_seed']);np.random.seed(p['evaluation_seed']);torch.manual_seed(p['evaluation_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=torch.device(device)
    require(d.type in ['cpu','cuda'],'Only CPU/CUDA are supported')
    info={'python':platform.python_version(),'torch':str(torch.__version__), 'numpy':np.__version__,
          'sentencepiece':importlib.metadata.version('sentencepiece'),'pyarrow':importlib.metadata.version('pyarrow'),
          'device_type':d.type,'precision':p['precision'],'sdpa':'math','deterministic':True,
          'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cpu_threads':p['cpu_threads'],
          'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if d.type=='cuda':
        require(torch.cuda.is_available(),'CUDA is unavailable in this Python environment')
        torch.cuda.set_device(d);torch.cuda.manual_seed_all(p['evaluation_seed'])
        props=torch.cuda.get_device_properties(d)
        info.update(gpu_name=props.name,gpu_uuid=str(getattr(props,'uuid',props.name)),gpu_memory=props.total_memory)
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                                  capture_output=True,text=True,check=True,timeout=10)
            info['driver_versions']=sorted(set(result.stdout.splitlines()))
        except (OSError,subprocess.SubprocessError):
            info['driver_versions']='unavailable'
        require(p['precision']!='bf16' or torch.cuda.is_bf16_supported(),'GPU does not support bf16')
    else:
        require(p['precision']=='fp32','For CPU, set protocol precision=fp32 BEFORE prepare; official server default is bf16')
    return info

def load_model(run,milestone,device):
    import torch
    from src.models.config import ModelConfig
    from src.models.transformer import build_model
    with checkpoint_bytes(run['checkpoints'][milestone]) as f:
        payload=torch.load(f,map_location='cpu',weights_only=True)
    expected={'run_id':run['run_id'],'pe_type':run['pe_method'],'init_seed':run['model_seed'],
              'nominal_tokens':run['milestones'][milestone]['nominal_tokens'],
              'actual_tokens':run['milestones'][milestone]['actual_tokens']}
    require(all(payload.get(k)==v for k,v in expected.items()),'Checkpoint payload does not match M3 run/milestone')
    model=build_model(ModelConfig.from_dict(run['config']))
    model.load_state_dict(payload['model_state_dict'],strict=True)
    del payload
    return model.to(device).eval()

def score_documents(model,cache,indices,kind,context,p,device):
    import torch
    import torch.nn.functional as F
    indices=np.asarray(indices,dtype=np.int64)
    require(len(indices)>0,'No eligible documents for evaluation')
    sums=np.zeros(len(indices),dtype=np.float64);counts=np.zeros(len(indices),dtype=np.int64)
    batch_size=p['batch_sizes'][str(context)];buckets={}
    start_time=time.monotonic();last_log=start_time;windows_done=0
    cuda=torch.device(device).type=='cuda'
    if cuda:torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    def flush(bucket):
        nonlocal windows_done
        batch=buckets.pop(bucket,[])
        if not batch:return
        size=max(len(t) for _,t,_ in batch)
        x=torch.full((len(batch),size),1,dtype=torch.long,device=device)
        y=torch.full_like(x,-100)
        for j,(_,tokens,first) in enumerate(batch):
            x[j,:len(tokens)]=torch.as_tensor(np.array(tokens,dtype=np.int64),device=device)
            y[j,first:len(tokens)]=x[j,first:len(tokens)]
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if cuda and p['precision']=='bf16' else contextlib.nullcontext()
        with torch.inference_mode(),amp:
            logits,_=model(x)
            losses=F.cross_entropy(logits[:,:-1].float().transpose(1,2),y[:,1:],ignore_index=-100,reduction='none')
            batch_sums=losses.double().sum(dim=1).cpu().numpy()
            batch_counts=(y[:,1:]!=-100).sum(dim=1).cpu().numpy()
        require(np.isfinite(batch_sums).all(),'Nonfinite model NLL')
        for j,(out,_,_) in enumerate(batch):sums[out]+=batch_sums[j];counts[out]+=batch_counts[j]
        windows_done+=len(batch)
    for out,i in enumerate(indices):
        tokens=cache[int(i)]
        windows=primary_windows(len(tokens),context,p['primary_stride']) if kind=='primary' else [length_window(len(tokens),context)]
        for begin,end,first in windows:
            bucket=(end-begin+63)//64
            buckets.setdefault(bucket,[]).append((out,tokens[begin:end],first))
            if len(buckets[bucket])>=batch_size:flush(bucket)
        if time.monotonic()-last_log>20:
            print(f'{kind} c{context}: {out+1:,}/{len(indices):,} documents; {windows_done:,} windows',flush=True)
            last_log=time.monotonic()
    for bucket in list(buckets):flush(bucket)
    expected=cache.lengths[indices]-1 if kind=='primary' else np.full(len(indices),256)
    require(np.array_equal(counts,expected),'Wrong target count: check shift, masking or windows')
    nll=float(sums.sum()/counts.sum())
    metrics={'nll':nll,'ppl':math.exp(nll),'token_count':int(counts.sum()),'document_count':len(indices),
             'elapsed_seconds':time.monotonic()-start_time,'windows':windows_done,
             'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(device) if cuda else 0}
    return {'document_id':cache.ids[indices],'nll_sum':sums,'token_count':counts},metrics
