#!/usr/bin/env python
"""Download the pinned public ESM checkpoints into the configured HF cache."""
import os
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download
HERE=Path(__file__).resolve().parent
if not os.environ.get('HF_HOME'):
    raise RuntimeError('Set HF_HOME explicitly; recommended: /data/users/akolchina/.cache/huggingface')
models=pd.read_csv(HERE/'models.csv')
for row in models[models.status.eq('ready_to_submit')].drop_duplicates(['checkpoint','revision']).itertuples(index=False):
    print(f'Caching {row.checkpoint}@{row.revision}',flush=True)
    snapshot_download(row.checkpoint,revision=row.revision)
