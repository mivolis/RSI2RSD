import copy
import unittest
import torch
import numpy as np
import c4
import forks
from test_c4 import tiny
from pathlib import Path
import tempfile,json
from types import SimpleNamespace
from unittest.mock import patch

class ForkContracts(unittest.TestCase):
    def test_schedule_crosses_domain_and_pairs(self):
        pools=c4.split_ids(np.repeat(np.arange(100),100))
        a=forks.schedule(14,93,3,101,pools)
        self.assertEqual([(x['block'],x['batch'],x['cycle']) for x in a],[(14,93,0),(15,0,1),(15,1,1)])
        self.assertEqual(len(a[0]['ids']),48)
        self.assertEqual(a,forks.schedule(14,93,3,101,pools))
        self.assertEqual(a[1:],forks.schedule(14,94,2,101,pools))
    def test_complete_rollback_and_continuation(self):
        torch.set_num_threads(2);c4.seed_all(5)
        a=c4.Learner(tiny(),1,tau=0);x=torch.rand(8,3,32,32)
        a.step(x,x);state=forks.snapshot(a)
        expected=a.step(x,x)['student_after'].clone()
        forks.restore(a,state)
        got=a.step(x,x)['student_after']
        self.assertTrue(torch.equal(got,expected))
        forks.restore(a,state)
        for name in ['student','teacher']:
            for k,v in state[name].items():self.assertTrue(torch.equal(v,getattr(a,name).state_dict()[k]))
        self.assertTrue(a.opt.state)

    def test_full_fork_raw_artifacts_and_identity(self):
        import robustbench.utils
        torch.set_num_threads(2);c4.seed_all(9)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';data.mkdir();parent=root/'parent';parent.mkdir()
            labels=np.tile(np.repeat(np.arange(100),100),5);np.save(data/'labels.npy',labels)
            for domain in c4.DOMAINS:
                a=np.lib.format.open_memmap(data/(domain+'.npy'),mode='w+',dtype='uint8',shape=(50000,32,32,3))
                a.flush();del a
            source=tiny()
            with torch.no_grad():source[-1].bias[0]=9.
            cfg=dict(domains=c4.DOMAINS,max_batches=0,gamma=1.,alpha=.999,seed=101)
            c4.atomic_json(parent/'config.json',cfg)
            dest=parent/'block_004/attempt_00';dest.mkdir(parents=True)
            learner=c4.Learner(source,1.);learner.save(dest/'checkpoint.pt',5,cfg)
            c4.atomic_json(dest/'complete.json',dict(files={'checkpoint.pt':c4.filehash(dest/'checkpoint.pt')}))
            args=SimpleNamespace(threads=2,parent=str(parent),after_block=5,output=str(root/'fork'),horizon=3,
                data=str(data),model_dir=str(root/'models'),device='cpu',stop_file='')
            with patch.object(robustbench.utils,'load_model',return_value=source):forks.run(args)
            out=root/'fork';result=json.loads((out/'comparison.json').read_text())
            self.assertEqual([r['h'] for r in result],[0,1,3])
            for r in result:self.assertAlmostEqual(r['I_feedback'],r['A_closed']-r['A_open'],places=10)
            # This synthetic source always selects class 0, so both target streams
            # are identical; an effect would indicate state leakage between arms.
            for i in range(1,4):
                with np.load(out/'accept_closed'/f'step_{i:04d}.npz') as cl,np.load(out/'accept_replay'/f'step_{i:04d}.npz') as ol:
                    self.assertTrue(np.array_equal(cl['pseudo'],ol['pseudo']))
                    self.assertTrue(np.array_equal(cl['mask'],ol['mask']))
                    self.assertTrue(np.array_equal(cl['student_after'],ol['student_after']))
            identity=json.loads((out/'reject_replay_identity.json').read_text())
            self.assertTrue(all(x['max_abs_logit_difference']==0 for x in identity))
            for name,h in json.loads((out/'complete.json').read_text())['files'].items():
                self.assertEqual(c4.filehash(out/name),h)
            import verify_forks
            verified=verify_forks.verify(out)
            self.assertEqual(verified['raw_future_records'],9)
            self.assertEqual(verified['pseudo_label_switches'],0)
            # Detect altered retained evidence rather than trusting aggregate JSON.
            (out/'comparison.json').write_text('[]')
            with self.assertRaises(AssertionError):verify_forks.verify(out)

if __name__=='__main__':unittest.main(verbosity=2)
