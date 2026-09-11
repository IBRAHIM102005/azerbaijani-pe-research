"""Evaluation content identities, durable commits and single-process locks."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / 'results/manifests/m3_handoff'
PES = ['learned', 'sinusoidal', 'rope', 'alibi', 'nope']
SEEDS = [17, 42, 1234, 2027, 5003]
MILESTONES = ['5m', '10m', '20m', '50m']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def check_file(path, expected):
    require(Path(path).is_file(), f'Missing file: {path}')
    require(sha(path) == expected, f'SHA-256 mismatch: {path}')

@contextlib.contextmanager
def atomic_file(path, mode='wb'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, mode) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def write_json(path, value):
    with atomic_file(path, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        f.write('\n')

def event(work, kind, **detail):
    with (Path(work)/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'time': now(), 'event': kind, **detail})+'\n')
        f.flush()
        os.fsync(f.fileno())

@contextlib.contextmanager
def file_lock(path):
    """OS releases the lock after process/kernel termination; lock file may remain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        try:
            if os.name == 'nt':
                import msvcrt
                f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another Evaluation process holds {path}') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def source_identity():
    files = list((ROOT/'src/models').glob('*.py')) + list((ROOT/'src/evaluation').glob('*.py'))
    files += [ROOT/'scripts/evaluate.py', ROOT/'src/tokenizer/corpus.py', ROOT/'src/data/hashing.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}

def protocol():
    p = read_json(ROOT/'configs/evaluation_protocol.json')
    fixed = {'version':2, 'evaluation_seed':5, 'primary_context':512, 'primary_stride':256,
             'document_policy':'independent_documents_no_bos_append_one_eod',
             'primary_targets':'all_except_first_exactly_once_including_eod',
             'length_contexts':[512,1024,2048], 'length_anchor':2048, 'length_targets':256,
             'length_cohort':'all_documents_at_least_2048_tokens_including_eod',
             'learned_above_512':'not_applicable', 'workers_per_gpu':1,
             'bootstrap_resamples':10000, 'bootstrap_seed':5, 'confidence_level':0.95,
             'pairwise_comparisons':'all_10_holm', 'minimum_consistent_seeds':4,
             'min_practical_delta_nll':0.01}
    require(all(p.get(k)==v for k,v in fixed.items()), 'Scientific protocol changed; review implementation before use.')
    require(p['precision'] in ['bf16','fp32'], 'precision must be bf16 or fp32')
    for key in ['cpu_threads','tokenizer_threads','bootstrap_batch_size']:
        require(type(p[key]) is int and p[key]>0, f'Invalid {key}')
    require(set(p['batch_sizes'])=={'512','1024','2048'} and
            all(type(v) is int and v>0 for v in p['batch_sizes'].values()), 'Invalid batch sizes')
    return p

def evidence(work, paths):
    return {str(Path(p).relative_to(work)):sha(p) for p in sorted(paths)}

def verify_evidence(work, records):
    for name, expected in records.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'Unsafe evidence path')
        check_file(Path(work)/p, expected)
