"""Prespecified C4 accept/reject x endogenous/replayed-supervision forks.

The reject lineage generates the entire tape before either accept lineage runs.
Labels are used only in measurement. No candidate is selected for its outcomes.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torchvision.transforms as T
from c4 import (Learner, DOMAINS, MODEL, seed_all, keyed_seed, split_ids, reference_ids,
                reference_images, fixed_audit, audit, tensor_images, strong_transform,
                atomic_json, atomic_npz, filehash, load_domain)


def snapshot(learner):
    return copy.deepcopy(dict(student=learner.student.state_dict(),teacher=learner.teacher.state_dict(),
                              optimizer=learner.opt.state_dict()))


def restore(learner,state):
    learner.student.load_state_dict(state['student']);learner.teacher.load_state_dict(state['teacher'])
    # Optimizer.load_state_dict may reuse same-device tensors. Never let a
    # continuation mutate the immutable initial snapshot of another branch.
    learner.opt.load_state_dict(copy.deepcopy(state['optimizer']))


def schedule(start_block,start_batch,horizon,seed,pools):
    items=[];block=start_block;batch=start_batch
    while len(items)<horizon:
        cycle,di=divmod(block,len(DOMAINS));domain=DOMAINS[di]
        order=np.random.default_rng(keyed_seed(seed,cycle,domain,'order')).permutation(pools['adapt'])
        for b in range(batch,math.ceil(len(order)/64)):
            items.append(dict(block=block,cycle=cycle,domain=domain,batch=b,
                ids=order[b*64:(b+1)*64].tolist(),view_seed=keyed_seed(seed,cycle,domain,b,'views')))
            if len(items)==horizon:break
        block+=1;batch=0
    return items


def views(data,item,device,transform):
    arr=load_domain(data,item['domain'])
    seed_all(item['view_seed'])
    x=tensor_images(arr,np.array(item['ids']),device)
    return T.RandomHorizontalFlip(.5)(x),transform(x)


def save_step(path,row,values,labels):
    arrays={k:v.detach().cpu().numpy() for k,v in values.items() if isinstance(v,torch.Tensor)}
    arrays.update(ids=np.array(row['ids']),labels=labels[np.array(row['ids'])],view_seed=np.array(row['view_seed']))
    atomic_npz(path,**arrays)
    return dict(**row,selected=values['selected'],updated=values['updated'],
                proxy_pre=values['proxy_pre'],proxy_post=values['proxy_post'],proxy_delta=values['proxy_delta'],
                wrong_selected=int(((arrays['pseudo']!=arrays['labels']) & arrays['mask']).sum()))


def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    parent=Path(args.parent);cfg=json.loads((parent/'config.json').read_text())
    assert cfg['domains']==DOMAINS and not cfg['max_batches']
    assert args.after_block in [5,15]
    marker=list((parent/f'block_{args.after_block-1:03d}').glob('attempt_*/complete.json'))
    assert len(marker)==1
    checkpoint=marker[0].parent/'checkpoint.pt'
    assert filehash(checkpoint)==json.loads(marker[0].read_text())['files']['checkpoint.pt']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'config.json').exists(),'use a new attempt directory; never overwrite a fork'
    fcfg=dict(version='c4-fork-v3',parent_config=cfg,parent_checkpoint_sha256=filehash(checkpoint),
              parent_marker_sha256=filehash(marker[0]),after_block=args.after_block,horizon=args.horizon,
              fork_code_sha256=filehash(__file__),seed=cfg['seed'],selection='first nonempty next-domain proposal',
              tape='reject lineage, generated before accept continuations',
              command=__import__('sys').argv)
    atomic_json(out/'config.json',fcfg)
    labels=np.load(Path(args.data)/'labels.npy')[:10000]
    pools=split_ids(labels);ids=reference_ids(pools,labels);panel=reference_images(args.data,ids)
    from robustbench.utils import load_model
    source=load_model(model_name=MODEL,dataset='cifar100',threat_model='corruptions',model_dir=args.model_dir).eval().to(args.device)
    learner=Learner(source,cfg['gamma'],alpha=cfg['alpha']);learner.restore(checkpoint,cfg)
    transform=strong_transform();start=time.monotonic()
    candidates=schedule(args.after_block,0,94,cfg['seed'],pools)
    for item in candidates:
        before=snapshot(learner)
        weak,strong=views(args.data,item,args.device,transform)
        values=learner.step(weak,strong)
        row=save_step(out/f"candidate_{item['batch']:03d}.npz",item,values,labels)
        if values['updated']:
            accepted=snapshot(learner);rejected=before;candidate=item;break
    else:
        atomic_json(out/'complete.json',dict(status='no_eligible_update',after_block=args.after_block))
        return
    atomic_json(out/'candidate.json',row)
    seq=schedule(candidate['block'],candidate['batch']+1,args.horizon,cfg['seed'],pools)
    atomic_json(out/'future_schedule.json',seq)
    measures={};anchor=None
    for name,state in [('reject',rejected),('accept',accepted)]:
        restore(learner,state)
        learner.save(out/(name+'_initial.pt'),0,fcfg)
        q,anchor=fixed_audit(learner,panel,ids,labels,args.device,out/(name+'_initial_q.npz'),anchor)
        arr=load_domain(args.data,candidate['domain'])
        current=audit(learner,arr,pools['audit'],labels,args.device,out/(name+'_initial_current.npz'))
        measures[name]={'q':q,'current_domain':current}
    atomic_json(out/'local_quality.json',measures)
    horizons=set([0,1,5,20,args.horizon])
    horizons.update(i+1 for i,item in enumerate(seq) if item['batch']==93)
    all_metrics={}
    # Full donor first, then the two accept continuations.
    for branch,state in [('reject_closed',rejected),('accept_closed',accepted),('accept_replay',accepted)]:
        dest=out/branch;dest.mkdir();restore(learner,state)
        initial=measures['reject' if branch=='reject_closed' else 'accept']['q']
        metrics={0:initial};rows=[]
        with (dest/'updates.jsonl').open('w',buffering=1) as log:
            for i,item in enumerate(seq,1):
                if args.stop_file and Path(args.stop_file).exists():
                    learner.save(dest/'interrupted.pt',i-1,fcfg)
                    atomic_json(out/'status.json',dict(status='interrupted',branch=branch,completed=i-1));return
                weak,strong=views(args.data,item,args.device,transform)
                replay=None
                if branch=='accept_replay':
                    with np.load(out/'reject_closed'/f'step_{i:04d}.npz') as tape:
                        replay={k:torch.from_numpy(tape[k].copy()) for k in ['target_probs','pseudo','mask','confidence']}
                values=learner.step(weak,strong,replay=replay)
                row=save_step(dest/f'step_{i:04d}.npz',item,values,labels)
                log.write(json.dumps(dict(h=i,**row))+'\n');log.flush();os.fsync(log.fileno());rows.append(row)
                if i in horizons:
                    q,_=fixed_audit(learner,panel,ids,labels,args.device,dest/f'q_{i:04d}.npz',anchor)
                    metrics[i]=q
                    learner.save(dest/f'checkpoint_{i:04d}.pt',i,fcfg)
            atomic_json(dest/'metrics.json',metrics)
        all_metrics[branch]=metrics
        atomic_json(out/'status.json',dict(status='running',completed_branch=branch,elapsed_seconds=time.monotonic()-start))
    # Validate reject/replay identity over the first short horizon, retaining errors.
    restore(learner,rejected);identity=[]
    for i,item in enumerate(seq[:min(20,len(seq))],1):
        weak,strong=views(args.data,item,args.device,transform)
        with np.load(out/'reject_closed'/f'step_{i:04d}.npz') as tape:
            replay={k:torch.from_numpy(tape[k].copy()) for k in ['target_probs','pseudo','mask','confidence']}
            got=learner.step(weak,strong,replay=replay)
            error=float(np.max(np.abs(got['student_after'].cpu().numpy()-tape['student_after'])))
            identity.append(dict(h=i,max_abs_logit_difference=error));assert error<=1e-6
    atomic_json(out/'reject_replay_identity.json',identity)
    comparison=[]
    for h in sorted(all_metrics['reject_closed']):
        r=all_metrics['reject_closed'][h]['student']['ce']
        c=all_metrics['accept_closed'][h]['student']['ce'];o=all_metrics['accept_replay'][h]['student']['ce']
        comparison.append(dict(h=h,reject_closed=r,reject_replay=r,accept_closed=c,accept_replay=o,
                               A_closed=c-r,A_open=o-r,I_feedback=c-o,
                               reject_replay_full=f'implied by deterministic tape; first {len(identity)} steps explicitly verified'))
    atomic_json(out/'comparison.json',comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name in ['reject_closed','accept_closed','accept_replay']:
        ax[0].plot([x['h'] for x in comparison],[x[name] for x in comparison],label=name)
    for key in ['A_closed','A_open','I_feedback']:
        ax[1].plot([x['h'] for x in comparison],[x[key] for x in comparison],label=key)
    ax[1].axhline(0,color='gray',lw=.7)
    for a in ax:a.set_xlabel('Future scheduled updates');a.legend(fontsize=8);a.grid(alpha=.2)
    ax[0].set_ylabel('Fixed-Q student CE');ax[1].set_ylabel('Paired risk contrast')
    fig.suptitle(f'C4-5: prespecified fork after block {args.after_block}; one parent, exploratory')
    fig.tight_layout();fig.savefig(out/'C4-5_feedback.png');plt.close(fig)
    hashes={str(p.relative_to(out)):filehash(p) for p in out.rglob('*') if p.is_file() and p.name not in ['complete.json','status.json']}
    atomic_json(out/'complete.json',dict(status='complete',horizon=args.horizon,files=hashes,
                                        elapsed_seconds=time.monotonic()-start))
    atomic_json(out/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-start))
    print(json.dumps(comparison),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--after-block',type=int,required=True,choices=[5,15])
    p.add_argument('--horizon',type=int,default=94);p.add_argument('--model-dir',default='checkpoints')
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--stop-file',default='STOP')
    args=p.parse_args()
    try:run(args)
    except Exception:
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/'failure.txt').write_text(traceback.format_exc());raise
