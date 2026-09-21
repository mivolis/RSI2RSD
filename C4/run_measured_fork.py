"""Run unchanged fork entry point and retain host/runtime memory provenance."""
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time

import torch

output=Path(sys.argv[1])
sys.argv=['forks.py']+sys.argv[2:]
started=time.time()
runpy.run_path('forks.py',run_name='__main__')
from c4 import atomic_json
atomic_json(output/'runtime.json',dict(
    host=platform.node(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    slurm_job_id=os.environ.get('SLURM_JOB_ID'),started_unix=started,finished_unix=time.time(),
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
