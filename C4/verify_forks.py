"""Independently recompute completed fork contrasts from retained raw predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from c4 import filehash, atomic_json


def verify(root):
    root = Path(root)
    marker = json.loads((root / 'complete.json').read_text())
    assert marker['status'] == 'complete', marker
    for name, digest in marker['files'].items():
        assert filehash(root / name) == digest, name
    config = json.loads((root / 'config.json').read_text())
    horizon = config['horizon']
    seq = json.loads((root / 'future_schedule.json').read_text())
    assert len(seq) == horizon
    initial = json.loads((root / 'local_quality.json').read_text())
    for decision in ['accept', 'reject']:
        for suffix, metrics_key in [('q', 'q'), ('current', 'current_domain')]:
            with np.load(root / f'{decision}_initial_{suffix}.npz') as ar:
                for model in ['student', 'teacher', 'anchor']:
                    ce = float(F.cross_entropy(torch.from_numpy(ar[model+'_logits']),
                               torch.as_tensor(ar['labels'], dtype=torch.long)))
                    assert abs(ce-initial[decision][metrics_key][model]['ce']) < 1e-5
    metrics = {}
    differences = {'pseudo_label_switches': 0, 'mask_switches': 0}
    for branch in ['reject_closed', 'accept_closed', 'accept_replay']:
        d = root / branch
        rows = [json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
        assert len(rows) == horizon == len(list(d.glob('step_*.npz')))
        metrics[branch] = json.loads((d/'metrics.json').read_text())
        for h, metric in metrics[branch].items():
            if int(h) == 0:
                decision = 'reject' if branch == 'reject_closed' else 'accept'
                assert metric == initial[decision]['q']
                continue
            with np.load(d/f'q_{int(h):04d}.npz') as ar:
                for model in ['student', 'teacher', 'anchor']:
                    ce = float(F.cross_entropy(torch.from_numpy(ar[model+'_logits']),
                               torch.as_tensor(ar['labels'],dtype=torch.long)))
                    assert abs(ce-metric[model]['ce']) < 1e-5
        for i, (row, item) in enumerate(zip(rows, seq), 1):
            with np.load(d/f'step_{i:04d}.npz') as ar:
                assert row['h'] == i and row['ids'] == item['ids']
                assert ar['ids'].tolist() == item['ids']
                assert int(ar['view_seed']) == item['view_seed']
                assert int(ar['mask'].sum()) == row['selected']
                if branch != 'reject_closed':
                    with np.load(root/'reject_closed'/f'step_{i:04d}.npz') as tape:
                        if branch == 'accept_replay':
                            for key in ['target_probs', 'pseudo', 'mask', 'confidence']:
                                assert np.array_equal(ar[key], tape[key]), (i,key)
                        else:
                            differences['pseudo_label_switches'] += int((ar['pseudo'] != tape['pseudo']).sum())
                            differences['mask_switches'] += int((ar['mask'] != tape['mask']).sum())
    comp = json.loads((root/'comparison.json').read_text())
    for row in comp:
        h = str(row['h'])
        r,c,o = [metrics[b][h]['student']['ce'] for b in
                 ['reject_closed','accept_closed','accept_replay']]
        for key,value in dict(A_closed=c-r,A_open=o-r,I_feedback=c-o).items():
            assert abs(row[key]-value) < 1e-10
    identity = json.loads((root/'reject_replay_identity.json').read_text())
    assert len(identity) == min(horizon,20)
    assert max(x['max_abs_logit_difference'] for x in identity) <= 1e-6
    result = dict(status='verified',horizon=horizon,hashed_files=len(marker['files']),
                  raw_future_records=3*horizon,**differences,
                  reject_replay_explicit_steps=len(identity),
                  scientific_scope='one parent; finite exploratory panel; no stability certificate')
    atomic_json(root/'verification.json', result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();verify(a.root)
