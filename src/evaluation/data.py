"""M1-compatible tokenization; document-preserving, paired evaluation windows."""
from __future__ import annotations
from itertools import zip_longest
from pathlib import Path
import os
import shutil
import tempfile
import time
import numpy as np
from .common import require,check_file,read_json,write_json,sha,digest

class DocumentCache:
    def __init__(self, path):
        self.path=Path(path)
        self.meta=read_json(self.path/'metadata.json')
        for n,h in self.meta['files'].items(): check_file(self.path/n,h)
        with np.load(self.path/'index.npz',allow_pickle=False) as z:
            self.ids=z['document_id'];self.offsets=z['offsets'];self.sources=z['source']
        self.tokens=np.memmap(self.path/'tokens.bin',mode='r',dtype='<u2')
        self.lengths=np.diff(self.offsets)
        require(len(self.ids)==len(self.lengths) and len(self.tokens)==self.offsets[-1], 'Invalid cache index')
    def __len__(self): return len(self.ids)
    def __getitem__(self,i): return self.tokens[self.offsets[i]:self.offsets[i+1]]

def build_cache(work,w,split,allow_test=False):
    require(split in ['validation','test'],'Invalid split')
    require(split!='test' or allow_test,'Held-out test cache requires final-test gate')
    spec=w['data'][split];identity=digest({'data':spec,'tokenizer':w['tokenizer'],'policy':w['protocol']['document_policy']})
    path=Path(work)/'cache'/split
    for a in spec.values():check_file(a['path'],a['sha256'])
    check_file(w['tokenizer']['path'],w['tokenizer']['sha256'])
    if (path/'metadata.json').exists():
        cache=DocumentCache(path)
        require(cache.meta['identity']==identity,'Cache data identity changed')
        return cache
    import pyarrow.parquet as pq
    import sentencepiece as spm
    from src.tokenizer.corpus import tokenizer_text
    sp=spm.SentencePieceProcessor(model_file=w['tokenizer']['path'])
    require(sp.vocab_size()==16000 and sp.eos_id()==1 and sp.bos_id()==-1 and sp.pad_id()==-1,
            'Unexpected frozen tokenizer special tokens')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=split+'-',dir=path.parent))
    cols=['document_id','source','source_group','canonical_text_hash']
    corpus=pq.ParquetFile(spec['processed_corpus']['path'])
    manifest=pq.ParquetFile(spec['manifests']['path'])
    require(corpus.metadata.num_rows==manifest.metadata.num_rows,'Manifest/corpus row counts differ')
    ids=[];sources=[];offsets=[0];last_id='';last_log=time.monotonic()
    try:
        with (temp/'tokens.bin').open('wb') as f:
            a=corpus.iter_batches(batch_size=1024,columns=cols+['text','split'])
            b=manifest.iter_batches(batch_size=1024,columns=cols+['processed_row'])
            for cb,mb in zip_longest(a,b):
                require(cb is not None and mb is not None,'Manifest stream mismatch')
                c=cb.to_pydict();m=mb.to_pydict()
                require(all(c[k]==m[k] for k in cols),'Manifest rows do not identify corresponding corpus documents')
                require(all(s==split for s in c['split']),'Wrong split in corpus')
                require(m['processed_row']==list(range(len(ids),len(ids)+len(c['text']))),'Unexpected processed_row order')
                encoded=sp.encode([tokenizer_text(t) for t in c['text']],out_type=int,
                                  num_threads=w['protocol']['tokenizer_threads'])
                for doc_id,source,tokens in zip(c['document_id'],c['source'],encoded):
                    require(len(doc_id)==64 and doc_id>last_id,'Document IDs must be unique and sorted')
                    require(len(tokens)>0 and max(tokens)<16000,'Empty/invalid tokenized document')
                    last_id=doc_id
                    tokens.append(1)
                    f.write(np.asarray(tokens,dtype='<u2').tobytes())
                    offsets.append(offsets[-1]+len(tokens));ids.append(doc_id);sources.append(source)
                if time.monotonic()-last_log>20:
                    print(f'{split} cache: {len(ids):,}/{corpus.metadata.num_rows:,} documents',flush=True)
                    last_log=time.monotonic()
        require(len(ids)==corpus.metadata.num_rows and len(ids)>0,'Empty/incomplete split')
        np.savez_compressed(temp/'index.npz',document_id=np.asarray(ids,dtype='S64'),
                            source=np.asarray(sources),offsets=np.asarray(offsets,dtype=np.int64))
        meta={'identity':identity,'split':split,'documents':len(ids),'tokens_including_eod':offsets[-1],
              'primary_targets':offsets[-1]-len(ids),
              'length_documents':int(np.count_nonzero(np.diff(offsets)>=2048)),
              'files':{n:sha(temp/n) for n in ['tokens.bin','index.npz']}}
        write_json(temp/'metadata.json',meta)
        require(not path.exists(),'Uncommitted cache directory exists; inspect it before retrying')
        os.replace(temp,path)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print(f'Cached {split}: {len(ids):,} documents; {meta["primary_targets"]:,} primary targets',flush=True)
    return DocumentCache(path)

def primary_windows(n,context=512,stride=256):
    """Yield (start, end, local_first_target); every target index 1..n-1 once."""
    require(1<=stride<context and n>=2,'Invalid primary window parameters')
    end=min(context,n);previous=1
    while True:
        start=max(0,end-context)
        yield start,end,previous-start
        if end==n:return
        previous=end;end=min(n,end+stride)

def length_window(n,context,anchor=2048,targets=256):
    require(n>=anchor and targets<context<=anchor,'Invalid matched length window')
    start=anchor-context
    return start,anchor,anchor-targets-start
