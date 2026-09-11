"""Evaluation public validation interface; final test is controlled by scripts/evaluate.py."""
def evaluate(model, manifest_path, device='cpu'):
    """Return {nll, token_count, ...} on frozen validation only.

    Set EVALUATION_WORK_DIR to the prepared workspace. Official matrix/final-test
    artifacts must be produced by the gated CLI, which verifies checkpoints.
    """
    import os
    from pathlib import Path
    import numpy as np
    from .common import require,sha
    from .inputs import load_workspace
    from .data import build_cache
    from .metrics import configure_runtime,score_documents
    work=os.environ.get('EVALUATION_WORK_DIR')
    require(work is not None,'Set EVALUATION_WORK_DIR to your prepared Evaluation workspace')
    w=load_workspace(work,require_weights=False)
    require(sha(Path(manifest_path))==w['data']['validation']['manifests']['sha256'],
            'This public adapter supports frozen validation only. Use final-test for the held-out test batch.')
    configure_runtime(w['protocol'],device)
    cache=build_cache(work,w,'validation')
    _,metrics=score_documents(model.to(device),cache,np.arange(len(cache)),'primary',512,w['protocol'],device)
    return metrics
