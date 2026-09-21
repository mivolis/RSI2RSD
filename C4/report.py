"""Report committed C4 blocks; incomplete attempts are retained but never pooled."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from c4 import atomic_json, DOMAINS, filehash


def report(root, verify=False, verify_only=False):
    assert not verify_only or verify
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    arms={};rows=[];updates=[]
    for cfg in sorted(root.glob('gamma*/config.json')):
        config=json.loads(cfg.read_text());name=cfg.parent.name
        blocks=[];local=[]
        for marker in sorted(cfg.parent.glob('block_*/attempt_*/complete.json')):
            m=json.loads(marker.read_text())
            if verify:
                for filename,sha in m['files'].items():
                    assert filehash(marker.parent/filename)==sha,(marker,filename)
            b=json.loads((marker.parent/'summary.json').read_text())
            logs=[json.loads(x) for x in (marker.parent/'updates.jsonl').read_text().splitlines()]
            assert len(logs)==b['batches']
            assert len(list(marker.parent.glob('batch_*.npz')))==len(logs)
            assert sum(z['n'] for z in logs)==b['images']
            assert sum(z['selected'] for z in logs)==b['selected']
            # Aggregated values are independently checked from the per-sample archive.
            if verify:
                if 'fixed_q' in b:
                    import torch
                    import torch.nn.functional as F
                    with np.load(marker.parent/'fixed_q.npz') as q:
                        y=torch.as_tensor(q['labels'],dtype=torch.long)
                        for model in ['student','teacher','anchor']:
                            ce=float(F.cross_entropy(torch.from_numpy(q[model+'_logits']),y))
                            assert abs(ce-b['fixed_q'][model]['ce'])<1e-5
                for row in logs:
                    with np.load(marker.parent/f"batch_{row['batch']:03d}.npz") as a:
                        assert int(a['mask'].sum())==row['selected']
                        assert int((a['pseudo']!=a['labels']).sum())==row['wrong_all']
                        if row['selected']:
                            mask=a['mask'];y=a['pseudo'][mask]
                            import torch
                            import torch.nn.functional as F
                            pre=float(F.cross_entropy(torch.from_numpy(a['student_before'][mask]),torch.from_numpy(y)))
                            post=float(F.cross_entropy(torch.from_numpy(a['student_after'][mask]),torch.from_numpy(y)))
                            assert abs(pre-row['proxy_pre'])<1e-5
                            assert abs(post-row['proxy_post'])<1e-5
            blocks.append(b);rows.append(dict(arm=name,**b))
            local.extend(z for z in logs if z['local_probe'])
            updates.extend(dict(arm=name,**z) for z in logs)
        assert len({b['block'] for b in blocks})==len(blocks)
        arms[name]=(config,blocks,local)
    if verify_only:
        print(json.dumps(dict(status='verified',blocks=len(rows),batches=len(updates),
                              report_files_unchanged=True)),flush=True)
        return
    atomic_json(out/'blocks.json',rows)
    with (out/'updates.jsonl').open('w') as f:
        for row in updates:f.write(json.dumps(row,allow_nan=False)+'\n')
    if not rows:
        (out/'README.md').write_text('# C4 pilot\n\nNo complete domain blocks yet.\n')
        return
    plt.rcParams.update({'font.size':9,'figure.dpi':130})
    fig,ax=plt.subplots(figsize=(6,4));quadrants={}
    tol=1e-6
    for name,(_,blocks,local) in arms.items():
        points=[z for z in local if z['proxy_delta'] is not None]
        x=np.array([z['proxy_delta'] for z in points]);y=np.array([z['local_delta'] for z in points])
        ax.scatter(x,y,s=16,alpha=.7,label=name)
        quadrants[name]={'proxy_down_local_down':int(((x < -tol)&(y < -tol)).sum()),
            'proxy_down_local_up':int(((x < -tol)&(y > tol)).sum()),
            'proxy_up_local_down':int(((x > tol)&(y < -tol)).sum()),
            'proxy_up_local_up':int(((x > tol)&(y > tol)).sum()),
            'near_zero':int(((np.abs(x)<=tol)|(np.abs(y)<=tol)).sum()),
            'empty_mask_probes':sum(z['proxy_delta'] is None for z in local)}
    ax.axhline(0,c='gray',lw=.7);ax.axvline(0,c='gray',lw=.7)
    ax.set(xlabel='Same-target proxy CE change',ylabel='Held-out local probe CE change',
           title='C4-1: sampled local updates (finite-sample measurements)')
    ax.legend();fig.tight_layout();fig.savefig(out/'C4-1_local.png');plt.close(fig)
    atomic_json(out/'local_sign_counts.json',{'tolerance':tol,'arms':quadrants})
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for name,(_,blocks,_) in arms.items():
        if not blocks:continue
        ts=np.array([b['block'] for b in blocks])+1
        axes[0,0].plot(ts,[b['post']['student']['error'] for b in blocks],label=name)
        axes[0,0].plot(ts,[b['post']['teacher']['error'] for b in blocks],ls=':',alpha=.7,label=name+' teacher')
        axes[0,1].plot(ts,[b['post']['student']['ce'] for b in blocks],label=name)
        axes[1,0].plot(ts,[b['wrong_all']/b['images'] for b in blocks],label=name+' all')
        axes[1,0].plot(ts,[b['wrong_selected']/b['selected'] if b['selected'] else np.nan for b in blocks],ls=':',label=name+' selected')
        axes[1,1].plot(ts,[b['selected']/b['images'] for b in blocks],label=name)
        concentrations=[]
        for b in blocks:
            v=np.array(b['counts_selected']);v=v[v>0]
            concentrations.append(1+np.sum(v/v.sum()*np.log(v/v.sum()))/np.log(100) if len(v) else np.nan)
        axes[2,0].plot(ts,concentrations,label=name)
        axes[2,1].plot(ts,[b['disagreements']/b['images'] for b in blocks],label=name)
    ref=max((v[1] for v in arms.values()),key=len)
    axes[0,0].plot([b['block']+1 for b in ref],[b['pre']['anchor']['error'] for b in ref],c='black',ls='--',label='Frozen source')
    axes[0,1].plot([b['block']+1 for b in ref],[b['pre']['anchor']['ce'] for b in ref],c='black',ls='--',label='Frozen source')
    titles=['Error (solid student, dotted teacher)','Student CE','Pseudo-label error','Selection coverage','Selected-class concentration','Teacher/student disagreement']
    for ax,title in zip(axes.ravel(),titles):
        ax.set_title(title);ax.grid(alpha=.15);ax.legend(fontsize=6)
        for t in range(1,len(ref)+1):ax.axvline(t,c='gray',alpha=.12,lw=.5)
        ax.set_xlabel('Domain block (15 blocks per cycle)')
    fig.suptitle('C4-2: complete committed trajectories; domain difficulty varies',fontsize=12)
    fig.tight_layout();fig.savefig(out/'C4-2_trajectories.png');plt.close(fig)
    fig,axes=plt.subplots(5,3,figsize=(12,13),sharex=True)
    for ax,domain in zip(axes.ravel(),DOMAINS):
        for name,(_,blocks,_) in arms.items():
            ds=[b for b in blocks if b['domain']==domain]
            ax.plot([b['cycle']+1 for b in ds],[b['post']['student']['error'] for b in ds],'-o',label=name+' post',ms=3)
            ax.plot([b['cycle']+1 for b in ds],[b['pre']['student']['error'] for b in ds],'--x',label=name+' pre',ms=3)
        ax.set_title(domain);ax.set_xlabel('Visit');ax.set_ylabel('Student error');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=6)
    fig.suptitle('C4-3: same-corruption returns; all 15 domains',fontsize=12)
    fig.tight_layout();fig.savefig(out/'C4-3_returns.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7));summary=[]
    for name,(cfg,blocks,_) in arms.items():
        if not blocks:continue
        first={b['domain']:b for b in blocks if b['cycle']==0}
        last={b['domain']:b for b in blocks}
        changes=[b['post']['student']['error']-first[d]['post']['student']['error']
                 for d,b in last.items() if d in first and b['cycle']>0]
        final_cycle=max(b['cycle'] for b in blocks)
        vals=[np.mean([b['post']['student']['error'] for b in blocks]),
              np.mean(changes) if changes else np.nan,
              np.mean([b['post']['student']['error'] for b in blocks if b['cycle']==final_cycle]),
              sum(b['selected'] for b in blocks)/sum(b['images'] for b in blocks),
              sum(b['optimizer_steps'] for b in blocks),
              sum(b['seconds_including_checkpoint'] for b in blocks)/60]
        for ax,val in zip(axes.ravel(),vals):ax.scatter(cfg['gamma'],val,label=name)
        summary.append(dict(arm=name,gamma=cfg['gamma'],committed_blocks=len(blocks),
                            intended_blocks=cfg['cycles']*len(cfg['domains']),
                            mean_post_error=float(vals[0]),mean_return_change=float(vals[1]) if changes else None,
                            coverage=float(vals[3]),optimizer_steps=int(vals[4]),elapsed_block_minutes=float(vals[5])))
    for ax,title in zip(axes.ravel(),['Mean post-block error','Mean same-domain return change',
                'Latest cycle error (may be partial)','Selection coverage','Actual optimizer steps','Block time (minutes)']):
        ax.set(title=title,xlabel='gamma');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('C4-4: exploratory endpoint comparison, one seed',fontsize=12)
    fig.tight_layout();fig.savefig(out/'C4-4_sweep.png');plt.close(fig)
    atomic_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,(cfg,blocks,_) in arms.items():
        initial=Path(root/name/'fixed_q_initial.json')
        if not initial.exists():continue
        q0=json.loads(initial.read_text())
        usable=[b for b in blocks if 'fixed_q' in b]
        xx=[0]+[b['block']+1 for b in usable]
        for ax,metric in zip(axes,['ce','error']):
            ax.plot(xx,[q0['student'][metric]]+[b['fixed_q']['student'][metric] for b in usable],label=name)
            ax.plot(xx,[q0['teacher'][metric]]+[b['fixed_q']['teacher'][metric] for b in usable],ls=':',label=name+' teacher')
            ax.axhline(q0['anchor'][metric],color='black',ls='--',alpha=.4)
            ax.set(xlabel='Completed domain blocks',ylabel='Fixed-Q '+metric);ax.legend(fontsize=7)
    fig.suptitle('C4-2b: fixed reference distribution, same 1,500 image-corruption pairs')
    fig.tight_layout();fig.savefig(out/'C4-2b_fixed_reference.png');plt.close(fig)
    text=['# C4 meeting pilot: current observations','',
          'Exploration only. One seed; no causal feedback or phase-transition conclusion.',
          'Partial arms can cover different domains: compare matched completed blocks only.','',
          '| Arm | Blocks | Mean post error | Same-domain return change | Selection coverage | Updates |',
          '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        delta='pending' if s['mean_return_change'] is None else f"{s['mean_return_change']:+.4f}"
        text.append(f"| {s['arm']} | {s['committed_blocks']}/{s['intended_blocks']} | {s['mean_post_error']:.4f} | {delta} | {s['coverage']:.4f} | {s['optimizer_steps']} |")
    text+=['','Positive return change means higher error; first-versus-last post-block visits.',
           'Full raw logs, predictions and checkpoints remain in the corresponding run directories.','']
    for f in ['C4-1_local.png','C4-2_trajectories.png','C4-2b_fixed_reference.png','C4-3_returns.png','C4-4_sweep.png']:
        text.append(f'![{f}]({f})\n')
    (out/'README.md').write_text('\n'.join(text))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--verify',action='store_true')
    p.add_argument('--verify-only',action='store_true',help='validate without rewriting retained report artifacts')
    a=p.parse_args();report(a.root,a.verify or a.verify_only,a.verify_only)
