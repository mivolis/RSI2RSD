"""Two scheduler-assigned GPUs; host validation precedes the full-cycle extension."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

from c4 import atomic_json

root=Path(__file__).resolve().parent
os.chdir(root)
job=os.environ['SLURM_JOB_ID']
gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(gpus)==2, 'respect exactly two scheduler-assigned GPUs'
out=root/'migration_results'/('job'+job)
out.mkdir(parents=True,exist_ok=True)
assert not (out/'status.json').exists(), 'new attempt required'
start=time.time()
atomic_json(out/'status.json',dict(status='starting',job=job))
for horizon in [20,94,1410]:
    if horizon==1410:
        durations=[json.loads((out/f'after{b}_h94/complete.json').read_text())['elapsed_seconds'] for b in [5,15]]
        estimate=max(durations)*(1410/94)*1.5+180
        if os.environ.get('SLURM_JOB_END_TIME'):
            end=float(os.environ['SLURM_JOB_END_TIME'])
        else:
            info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
            end_text=next(x.split('=',1)[1] for x in info.split() if x.startswith('EndTime='))
            end=datetime.fromisoformat(end_text).timestamp()
        remaining=end-time.time()
        atomic_json(out/'long_horizon_budget.json',dict(estimated_seconds=estimate,remaining_seconds=remaining,
                    estimate_basis='measured H94 runtime x horizon ratio x 1.5 plus 180 seconds'))
        if remaining<estimate:
            atomic_json(out/'status.json',dict(status='short_forks_complete_long_deferred',reason='walltime margin',job=job))
            sys.exit(3)
    processes=[]
    for block,gpu in zip([5,15],gpus):
        dest=out/f'after{block}_h{horizon}'
        log=(out/f'after{block}_h{horizon}.log').open('w')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        cmd=[sys.executable,'-u','run_measured_fork.py',str(dest),
             '--parent','results/gamma1','--data','data/CIFAR-100-C',
             '--after-block',str(block),'--horizon',str(horizon),'--output',str(dest)]
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((block,p,log,dest))
    codes=[]
    for block,p,log,dest in processes:
        code=p.wait();log.close();codes.append(code)
    atomic_json(out/f'h{horizon}_exits.json',dict(after5=codes[0],after15=codes[1]))
    if any(codes):
        atomic_json(out/'status.json',dict(status='failed',horizon=horizon,exits=codes,job=job))
        sys.exit(1)
    for block,p,log,dest in processes:
        with (out/f'verify_after{block}_h{horizon}.log').open('w') as v:
            subprocess.run([sys.executable,'verify_forks.py',str(dest)],stdout=v,stderr=subprocess.STDOUT,check=True)
    atomic_json(out/'status.json',dict(status='running',completed_horizon=horizon,job=job,elapsed_seconds=time.time()-start))
atomic_json(out/'status.json',dict(status='complete',job=job,elapsed_seconds=time.time()-start))
