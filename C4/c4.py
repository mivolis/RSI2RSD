"""C4 natural CTTA; labels are consumed only by the measurement layer."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

DOMAINS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
           'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
           'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
ROOT = Path(__file__).resolve().parent
MODEL = 'Hendrycks2020AugMix_ResNeXt'
VERSION = 'c4-integrated-v3'


def filehash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(p, obj):
    p = Path(p)
    temp = p.with_name(p.name + '.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)


def atomic_npz(p, **arrays):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def keyed_seed(seed, *parts):
    key = '|'.join(map(str, (seed,) + parts))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')


def split_ids(labels):
    assert labels.shape == (10000,)
    assert np.array_equal(np.unique(labels), np.arange(100))
    rng = np.random.default_rng(1000)
    pools = {k: [] for k in ['adapt', 'gate', 'audit', 'confirmation']}
    for c in range(100):
        ids = np.flatnonzero(labels == c)
        assert len(ids) == 100
        ids = rng.permutation(ids)
        for name, a, b in [('adapt', 0, 60), ('gate', 60, 70),
                           ('audit', 70, 85), ('confirmation', 85, 100)]:
            pools[name].extend(ids[a:b].tolist())
    pools = {k: np.array(v, dtype=np.int64) for k, v in pools.items()}
    assert len(np.unique(np.concatenate(list(pools.values())))) == 10000
    return pools


def strong_transform():
    path = ROOT / 'vendor/cotta/cifar/my_transforms.py'
    spec = importlib.util.spec_from_file_location('c4_cotta_transforms', path)
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    # Exact CoTTA CIFAR defaults; only deprecated RandomAffine argument names adapted.
    return T.Compose([
        mt.Clip(0., 1.), mt.ColorJitterPro(brightness=[.6, 1.4], contrast=[.7, 1.3],
            saturation=[.5, 1.5], hue=[-.06, .06], gamma=[.7, 1.3]),
        T.Pad(16, padding_mode='edge'), T.RandomAffine(degrees=[-15, 15],
            translate=(1/16, 1/16), scale=(.9, 1.1),
            interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.GaussianBlur(5, sigma=[.001, .5]), T.CenterCrop(32),
        T.RandomHorizontalFlip(.5), mt.GaussianNoise(0, .005), mt.Clip(0., 1.)])


def tensor_images(images, ids, device):
    x = np.array(images[ids], copy=True)
    return torch.from_numpy(x).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.


@torch.no_grad()
def predictions(model, x):
    return model(x).detach()


def measured(logits, labels):
    y = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    return {'error': float((logits.argmax(1) != y).float().mean()),
            'ce': float(F.cross_entropy(logits, y)), 'n': len(labels)}


class Learner:
    def __init__(self, source, gamma, lr=.001, alpha=.999, tau=.9):
        self.student = copy.deepcopy(source).eval().requires_grad_(True)
        self.teacher = copy.deepcopy(source).eval().requires_grad_(False)
        self.anchor = source.eval().requires_grad_(False)
        self.opt = torch.optim.SGD(self.student.parameters(), lr=lr, momentum=.9,
                                   weight_decay=0., nesterov=False)
        self.gamma, self.alpha, self.tau = gamma, alpha, tau
        assert 0 <= gamma <= 1 and 0 <= alpha <= 1

    def step(self, weak, strong, replay=None):
        """No ground-truth labels or outcome measurements are accepted by this API."""
        with torch.no_grad():
            pa = self.anchor(weak).softmax(1)
            pt = self.teacher(weak).softmax(1)
            target = (1 - self.gamma) * pa + self.gamma * pt
            confidence, pseudo = target.max(1)
            mask = confidence > self.tau
            if replay is not None:
                # The external tape supplies both targets AND the selection rule.
                target = replay['target_probs'].to(weak.device).detach()
                pseudo = replay['pseudo'].to(weak.device).detach()
                mask = replay['mask'].to(weak.device).detach()
                confidence = replay['confidence'].to(weak.device).detach()
                assert target.shape == pa.shape and pseudo.shape == mask.shape == (len(weak),)
            disagreement = self.student(weak).argmax(1) != pt.argmax(1)
        if not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite target')
        self.opt.zero_grad(set_to_none=True)
        logits = self.student(strong)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite student logits')
        before = logits.detach().clone()
        n = int(mask.sum())
        pre = post = None
        if n:
            loss = F.cross_entropy(logits[mask], pseudo[mask])
            pre = float(loss.detach())
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in self.student.parameters()
                       if p.grad is not None):
                raise FloatingPointError('nonfinite gradient')
            self.opt.step()
            if not all(torch.isfinite(p).all() for p in self.student.parameters()):
                raise FloatingPointError('nonfinite parameter')
            with torch.no_grad():
                after = self.student(strong)
                post = float(F.cross_entropy(after[mask], pseudo[mask]))
                if not math.isfinite(post):
                    raise FloatingPointError('nonfinite post-update loss')
                for teacher_p, student_p in zip(self.teacher.parameters(), self.student.parameters()):
                    teacher_p.mul_(self.alpha).add_(student_p, alpha=1 - self.alpha)
        else:
            after = before
        return dict(anchor_probs=pa.detach(), teacher_probs=pt.detach(),
                    target_probs=target.detach(), pseudo=pseudo.detach(), mask=mask.detach(),
                    confidence=confidence.detach(), student_before=before,
                    student_after=after.detach(), disagreement=disagreement.detach(),
                    selected=n, proxy_pre=pre, proxy_post=post,
                    proxy_delta=None if not n else post-pre, updated=bool(n))

    def save(self, path, next_block, config):
        obj = dict(student=self.student.state_dict(), teacher=self.teacher.state_dict(),
                   optimizer=self.opt.state_dict(), next_block=next_block, config=config,
                   rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                   rng_torch=torch.get_rng_state(),
                   rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        tmp = Path(str(path) + '.tmp')
        with tmp.open('wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def restore(self, path, config):
        obj = torch.load(path, map_location='cpu', weights_only=False)
        assert obj['config'] == config, 'resume configuration mismatch'
        self.student.load_state_dict(obj['student'])
        self.teacher.load_state_dict(obj['teacher'])
        self.opt.load_state_dict(obj['optimizer'])
        random.setstate(obj['rng_python'])
        np.random.set_state(obj['rng_numpy'])
        torch.set_rng_state(obj['rng_torch'])
        if obj['rng_cuda']:
            torch.cuda.set_rng_state_all(obj['rng_cuda'])
        return obj['next_block']


@torch.no_grad()
def audit(learner, images, ids, labels, device, path):
    arrays = {'ids': ids, 'labels': labels[ids]}
    result = {}
    for name in ['student', 'teacher', 'anchor']:
        logits = torch.cat([predictions(getattr(learner, name), tensor_images(images, z, device)).cpu()
                            for z in np.array_split(ids, math.ceil(len(ids) / 128))])
        arrays[name + '_logits'] = logits.numpy()
        result[name] = measured(logits, labels[ids])
    atomic_npz(path, **arrays)
    return result


def reference_ids(pools, labels):
    """One prespecified base ID per class; no confirmation IDs are read."""
    return np.array([pools['audit'][labels[pools['audit']] == c].min()
                     for c in range(100)], dtype=np.int64)


def load_domain(data, domain):
    arr=np.load(Path(data)/(domain+'.npy'),mmap_mode='r')
    if arr.shape == (50000,32,32,3):return arr[40000:50000]
    assert arr.shape == (10000,32,32,3), 'expected original or verified severity-5 extraction'
    return arr


def reference_images(data, ids):
    panels = []
    for domain in DOMAINS:
        arr = load_domain(data,domain)
        panels.append(np.array(arr[ids], copy=True))
    return np.concatenate(panels)


@torch.no_grad()
def fixed_audit(learner, panel, ids, labels, device, path, anchor_logits=None):
    """Pure fixed-Q audit; return source logits to avoid repeated anchor inference."""
    ys = np.tile(labels[ids], len(DOMAINS))
    arrays = dict(base_ids=np.tile(ids,len(DOMAINS)), labels=ys,
                  domains=np.repeat(np.array(DOMAINS),len(ids)))
    metrics = {}
    for name in ['student','teacher','anchor']:
        if name == 'anchor' and anchor_logits is not None:
            logits = anchor_logits
        else:
            logits = torch.cat([predictions(getattr(learner,name),
                tensor_images(panel,z,device)).cpu()
                for z in np.array_split(np.arange(len(panel)),math.ceil(len(panel)/128))])
        arrays[name+'_logits'] = logits.numpy()
        metrics[name] = measured(logits, ys)
    ps=torch.from_numpy(arrays['student_logits']).softmax(1)
    pt=torch.from_numpy(arrays['teacher_logits']).softmax(1)
    pa=torch.from_numpy(arrays['anchor_logits']).softmax(1)
    mix=(1-learner.gamma)*pa+learner.gamma*pt
    mid=(ps+pt)/2
    js=(.5*(ps*(ps.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(1)
        +.5*(pt*(pt.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(1)).mean()
    mix_ce=float(-mix[torch.arange(len(ys)),torch.as_tensor(ys,dtype=torch.long)].clamp_min(1e-30).log().mean())
    metrics['diagnostic_state']=[metrics['student']['ce'],mix_ce,float(js),float(mix.max(1).values.mean())]
    arrays['mixture_probs']=mix.numpy()
    atomic_npz(path,**arrays)
    return metrics,torch.from_numpy(arrays['anchor_logits'])


def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    labels_full = np.load(data / 'labels.npy')
    labels = labels_full[:10000]
    if len(labels_full) == 50000:
        assert all(np.array_equal(labels, labels_full[k:k+10000]) for k in range(0, 50000, 10000))
    else:
        assert len(labels_full) == 10000
    pools = split_ids(labels)
    atomic_npz(out / 'split_ids.npz', **pools)
    from robustbench.utils import load_model
    source = load_model(model_name=MODEL, dataset='cifar100', threat_model='corruptions',
                        model_dir=args.model_dir).eval().to(args.device)
    beta=getattr(args,'beta',.001)
    learner = Learner(source, args.gamma, alpha=1-beta)
    transform = strong_transform()
    config = dict(version=VERSION, model=MODEL, gamma=args.gamma, seed=args.seed,
        cycles=args.cycles, severity=5, batch_size=64, lr=.001, momentum=.9,
        alpha=1-beta, beta=beta, feedback_vector={'lambda':args.gamma,'beta':beta},
        fixed_q_domains=DOMAINS, fixed_q_ids=reference_ids(pools,labels).tolist(),
        input_layout='severity5_extract' if np.load(data/(DOMAINS[0]+'.npy'),mmap_mode='r').shape[0]==10000 else 'original_5_severities',
        tau=.9, trainable='all; eval mode, BN buffers frozen',
        weak='batchwise RandomHorizontalFlip(.5)', strong=repr(transform),
        split_seed=1000, domains=DOMAINS[:args.domains], max_batches=args.max_batches,
        code_sha256=filehash(__file__), cotta_commit=subprocess.check_output(
            ['git', '-C', str(ROOT/'vendor/cotta'), 'rev-parse', 'HEAD'], text=True).strip())
    cfg_path = out / 'config.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == config
    else:
        atomic_json(cfg_path, config)
    if not (out / 'environment.json').exists():
        atomic_json(out / 'environment.json', dict(torch=torch.__version__,
            python=sys.version, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else args.device,
            source_files={str(p.relative_to(args.model_dir)): filehash(p)
                          for p in Path(args.model_dir).rglob('*.pt')},
            source_parameters=sum(p.numel() for p in source.parameters()),
            command=sys.argv, start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        (out / 'pip_freeze.txt').write_text(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True))
    done = sorted(out.glob('block_*/attempt_*/complete.json'))
    committed = [json.loads(p.read_text()) | {'dir': str(p.parent)} for p in done]
    assert len({z['block'] for z in committed}) == len(committed)
    start = 0
    if committed:
        last = max(committed, key=lambda z: z['block'])
        assert sorted(z['block'] for z in committed) == list(range(last['block'] + 1))
        start = learner.restore(Path(last['dir'])/'checkpoint.pt', config)
    elif (out/'initial.pt').exists():
        learner.restore(out/'initial.pt', config)
    else:
        learner.save(out/'initial.pt', 0, config)
    panel_ids=reference_ids(pools,labels)
    panel=reference_images(data,panel_ids)
    if not (out/'fixed_q_initial.json').exists():
        assert start == 0, 'initial fixed-Q measurement must precede adaptation'
        initial_q,anchor_logits=fixed_audit(learner,panel,panel_ids,labels,args.device,out/'fixed_q_initial.npz')
        atomic_json(out/'fixed_q_initial.json',initial_q)
    else:
        with np.load(out/'fixed_q_initial.npz') as ar:
            anchor_logits=torch.from_numpy(ar['anchor_logits'].copy())
    start_clock = time.monotonic()
    domains = config['domains']
    for t in range(start, args.cycles * len(domains)):
        if args.stop_file and Path(args.stop_file).exists():
            atomic_json(out/'status.json', dict(status='stopped_at_boundary',next_block=t))
            return
        block_clock = time.monotonic()
        cycle, index = divmod(t, len(domains))
        domain = domains[index]
        images = load_domain(data,domain)
        base = out / f'block_{t:03d}'
        base.mkdir(exist_ok=True)
        attempt = 0
        while (base / f'attempt_{attempt:02d}').exists():
            attempt += 1
        dest = base/f'attempt_{attempt:02d}'
        dest.mkdir()
        order_seed = keyed_seed(args.seed, cycle, domain, 'order')
        order = np.random.default_rng(order_seed).permutation(pools['adapt'])
        np.save(dest/'order.npy', order)
        audit_pre = audit(learner, images, pools['audit'], labels, args.device, dest/'audit_pre.npz')
        probe_ids = pools['audit'][np.random.default_rng(1001).permutation(len(pools['audit']))[:128]]
        probe_x = tensor_images(images, probe_ids, args.device)
        batches = list(range(0, len(order), 64))
        if args.max_batches:
            batches = batches[:args.max_batches]
        records = []
        with (dest/'updates.jsonl').open('w', buffering=1) as log:
            for b, offset in enumerate(batches):
                tick = time.monotonic()
                ids = order[offset:offset+64]
                augmentation_seed = keyed_seed(args.seed, cycle, domain, b, 'views')
                seed_all(augmentation_seed)
                x = tensor_images(images, ids, args.device)
                weak = T.RandomHorizontalFlip(.5)(x)
                strong = transform(x)
                local = b in [0, len(batches)-1]
                if local:
                    before_probe = predictions(learner.student, probe_x)
                values = learner.step(weak, strong)
                # Ground-truth joins occur only after the optimizer/EMA decision.
                pred = values['pseudo'].cpu().numpy()
                selected = values['mask'].cpu().numpy()
                count = np.bincount(pred[selected], minlength=100)
                positive = count[count > 0]
                concentration = float(1 + np.sum((positive / positive.sum()) *
                    np.log(positive / positive.sum()))/np.log(100)) if len(positive) else None
                arrays = {k: v.cpu().numpy() for k,v in values.items() if isinstance(v, torch.Tensor)}
                arrays.update(ids=ids, labels=labels[ids], augmentation_seed=np.array(augmentation_seed))
                row = dict(block=t, cycle=cycle, domain=domain, batch=b, gamma=args.gamma,
                    seed=args.seed, n=len(ids), selected=values['selected'],
                    wrong_all=int((pred != labels[ids]).sum()),
                    wrong_selected=int(((pred != labels[ids]) & selected).sum()),
                    counts_selected=count.tolist(), concentration=concentration,
                    disagreement=int(values['disagreement'].sum()),
                    confidence_mean=float(values['confidence'].mean()),
                    proxy_pre=values['proxy_pre'], proxy_post=values['proxy_post'],
                    proxy_delta=values['proxy_delta'], updated=values['updated'],
                    local_probe=local, augmentation_seed=augmentation_seed)
                if local:
                    after_probe = predictions(learner.student, probe_x)
                    bp, ap = measured(before_probe, labels[probe_ids]), measured(after_probe, labels[probe_ids])
                    row.update(local_pre=bp, local_post=ap, local_delta=ap['ce']-bp['ce'])
                    arrays.update(probe_ids=probe_ids, probe_labels=labels[probe_ids],
                        probe_before=before_probe.cpu().numpy(), probe_after=after_probe.cpu().numpy())
                atomic_npz(dest/f'batch_{b:03d}.npz', **arrays)
                row['seconds'] = time.monotonic()-tick
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                records.append(row)
        audit_post = audit(learner, images, pools['audit'], labels, args.device, dest/'audit_post.npz')
        fixed_q,_=fixed_audit(learner,panel,panel_ids,labels,args.device,dest/'fixed_q.npz',anchor_logits)
        # Buffers of all three models must retain the source checkpoint values.
        for model in [learner.student, learner.teacher]:
            for name, buf in model.named_buffers():
                assert torch.equal(buf, dict(source.named_buffers())[name]), f'buffer mutated: {name}'
        summary = dict(block=t, cycle=cycle, domain=domain, pre=audit_pre, post=audit_post,fixed_q=fixed_q,
            batches=len(records), images=sum(z['n'] for z in records),
            selected=sum(z['selected'] for z in records),
            wrong_all=sum(z['wrong_all'] for z in records),
            wrong_selected=sum(z['wrong_selected'] for z in records),
            disagreements=sum(z['disagreement'] for z in records),
            optimizer_steps=sum(z['updated'] for z in records),
            empty_masks=sum(not z['updated'] for z in records),
            counts_selected=np.sum([z['counts_selected'] for z in records],axis=0).tolist(),
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        atomic_json(dest/'summary.json', summary)
        learner.save(dest/'checkpoint.pt', t+1, config)
        summary['seconds_including_checkpoint'] = time.monotonic()-block_clock
        atomic_json(dest/'summary.json', summary)
        hashes = {p.name: filehash(p) for p in sorted(dest.iterdir()) if p.is_file()}
        atomic_json(dest/'complete.json', dict(block=t, files=hashes))
        atomic_json(out/'status.json', dict(status='running', completed_blocks=t+1,
            total_blocks=args.cycles*len(domains), last_block_seconds=summary['seconds_including_checkpoint'],
            elapsed_seconds=time.monotonic()-start_clock, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print(json.dumps({'completed_block':t,'domain':domain,'cycle':cycle,
            'seconds':summary['seconds_including_checkpoint'],'student':audit_post['student'],
            'selected':summary['selected'],'steps':summary['optimizer_steps']}),flush=True)
    atomic_json(out/'status.json', dict(status='complete', completed_blocks=args.cycles*len(domains),
        total_blocks=args.cycles*len(domains), elapsed_seconds=time.monotonic()-start_clock))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model-dir', default='checkpoints')
    p.add_argument('--output', required=True)
    p.add_argument('--gamma', type=float, required=True, choices=[0., .5, 1.])
    # p.add_argument('--gamma', type=float, required=True)
    p.add_argument('--beta', type=float, default=.001)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--cycles', type=int, default=2)
    p.add_argument('--domains', type=int, default=15)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--stop-file', default='STOP')
    args = p.parse_args()
    try:
        run(args)
    except Exception:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output)/'failure.txt').write_text(traceback.format_exc())
        raise
