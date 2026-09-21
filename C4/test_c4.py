import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import c4


def tiny():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 100)).eval()


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        c4.seed_all(12)
        self.x = torch.rand(8, 3, 32, 32)

    def test_update_targets_and_diagnostic(self):
        a = c4.Learner(tiny(), .5, tau=0.)
        source = copy.deepcopy(a.anchor.state_dict())
        old_teacher = copy.deepcopy(a.teacher.state_dict())
        r = a.step(self.x, self.x)
        self.assertEqual(r['selected'], 8)
        self.assertTrue(torch.equal(r['target_probs'], .5*r['anchor_probs']+.5*r['teacher_probs']))
        self.assertAlmostEqual(r['proxy_pre'], float(torch.nn.functional.cross_entropy(
            r['student_before'],r['pseudo'])), places=6)
        self.assertAlmostEqual(r['proxy_post'], float(torch.nn.functional.cross_entropy(
            r['student_after'],r['pseudo'])), places=6)
        self.assertTrue(any(not torch.equal(v,a.student.state_dict()[k]) for k,v in source.items()))
        for k,v in a.anchor.state_dict().items():
            self.assertTrue(torch.equal(v,source[k]))
        for k,v in a.teacher.named_parameters():
            expected = old_teacher[k]*.999 + a.student.state_dict()[k]*.001
            self.assertTrue(torch.allclose(v, expected, atol=1e-7, rtol=1e-6))
        for k,v in a.student.named_buffers():
            self.assertTrue(torch.equal(v,source[k]))

    def test_empty_mask_leaves_entire_state_unchanged(self):
        a = c4.Learner(tiny(), 1., tau=1.)
        old = copy.deepcopy(a.student.state_dict())
        teacher = copy.deepcopy(a.teacher.state_dict())
        r = a.step(self.x,self.x)
        self.assertFalse(r['updated'])
        self.assertIsNone(r['proxy_delta'])
        self.assertFalse(a.opt.state)
        for k,v in old.items(): self.assertTrue(torch.equal(v,a.student.state_dict()[k]))
        for k,v in teacher.items(): self.assertTrue(torch.equal(v,a.teacher.state_dict()[k]))

    def test_resume_exact_and_audit_pure(self):
        a = c4.Learner(tiny(),1.,tau=0.)
        a.step(self.x,self.x)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'checkpoint.pt'
            a.save(path,1,{'test':1})
            b=c4.Learner(copy.deepcopy(a.anchor),1.,tau=0.)
            self.assertEqual(b.restore(path,{'test':1}),1)
            ar=a.step(self.x,self.x);br=b.step(self.x,self.x)
            self.assertTrue(torch.equal(ar['student_after'],br['student_after']))
            before=copy.deepcopy(a.student.state_dict())
            imgs=np.random.randint(0,256,(8,32,32,3),dtype=np.uint8)
            rng=torch.get_rng_state().clone()
            c4.audit(a,imgs,np.arange(8),np.arange(8),'cpu',Path(td)/'audit.npz')
            for k,v in before.items():self.assertTrue(torch.equal(v,a.student.state_dict()[k]))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_split_isolation_and_paired_views(self):
        pools=c4.split_ids(np.repeat(np.arange(100),100))
        self.assertEqual([len(pools[k]) for k in ['adapt','gate','audit','confirmation']],
                         [6000,1000,1500,1500])
        transform=c4.strong_transform()
        c4.seed_all(c4.keyed_seed(101,0,'fog',0,'views')); a=transform(self.x)
        c4.seed_all(c4.keyed_seed(101,0,'fog',0,'views')); b=transform(self.x)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(a.shape,self.x.shape)

    def test_nonfinite_is_not_a_scientific_result(self):
        a=c4.Learner(tiny(),1.,tau=0.)
        with self.assertRaises(FloatingPointError):a.step(self.x*float('nan'),self.x)

    def test_gamma_zero_ignores_teacher_targets(self):
        source=tiny();a=c4.Learner(source,0.,tau=0.);b=c4.Learner(copy.deepcopy(source),0.,tau=0.)
        with torch.no_grad():
            for p in b.teacher.parameters(): p.add_(.2)
        ar=a.step(self.x,self.x);br=b.step(self.x,self.x)
        self.assertTrue(torch.equal(ar['target_probs'],br['target_probs']))
        self.assertTrue(torch.equal(ar['student_after'],br['student_after']))

    def test_replay_identity_and_cut_feedback(self):
        source=tiny()
        donor=c4.Learner(source,1.,tau=0.)
        replay=c4.Learner(copy.deepcopy(source),1.,tau=1.)
        for _ in range(3):
            tape=donor.step(self.x,self.x)
            got=replay.step(self.x,self.x,replay=tape)
            self.assertTrue(torch.equal(tape['student_after'],got['student_after']))
        # Modifying the focal teacher cannot change external targets or masks.
        with torch.no_grad():
            for p in replay.teacher.parameters():p.add_(3)
        tape=donor.step(self.x,self.x)
        got=replay.step(self.x,self.x,replay=tape)
        self.assertTrue(torch.equal(tape['student_after'],got['student_after']))
        self.assertEqual(got['selected'],8)

    def test_fixed_reference_panel_identity_and_purity(self):
        labels=np.repeat(np.arange(100,dtype=np.uint8),100)
        pools=c4.split_ids(labels);ids=c4.reference_ids(pools,labels)
        self.assertTrue(set(ids).issubset(set(pools['audit'])))
        for c in range(100):self.assertEqual(ids[c],pools['audit'][labels[pools['audit']]==c].min())
        learner=c4.Learner(tiny(),.5)
        before=copy.deepcopy(learner.student.state_dict());rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as td,patch.object(c4,'DOMAINS',c4.DOMAINS[:2]):
            imgs=np.zeros((200,32,32,3),dtype=np.uint8)
            metrics,anchor=c4.fixed_audit(learner,imgs,ids,labels,'cpu',Path(td)/'q.npz')
            self.assertEqual(metrics['student']['n'],200)
            self.assertAlmostEqual(metrics['student']['ce'],metrics['diagnostic_state'][1],places=5)
            self.assertAlmostEqual(metrics['diagnostic_state'][2],0.,places=6)
            with patch.object(learner.anchor,'forward',side_effect=AssertionError('anchor recomputed')):
                again,_=c4.fixed_audit(learner,imgs,ids,labels,'cpu',Path(td)/'again.npz',anchor)
            self.assertEqual(metrics,again)
        for k,v in before.items():self.assertTrue(torch.equal(v,learner.student.state_dict()[k]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_end_to_end_artifacts_and_boundary_resume(self):
        # Synthetic images/tiny model test software only; never scientific evidence.
        import robustbench.utils
        import report
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); data=root/'data';data.mkdir()
            np.save(data/'labels.npy',np.tile(np.repeat(np.arange(100,dtype=np.uint8),100),5))
            for domain in c4.DOMAINS[:2]:
                arr=np.lib.format.open_memmap(data/(domain+'.npy'),mode='w+',dtype='uint8',shape=(50000,32,32,3))
                arr.flush();del arr
            source=tiny()
            with torch.no_grad():
                source[-1].bias[0]=9.0  # Force nonempty masks in the artifact round-trip test.
            stop=root/'STOP';out=root/'results/gamma0'
            args=SimpleNamespace(threads=2,seed=101,output=str(out),data=str(data),
                device='cpu',model_dir=str(root/'checkpoints'),gamma=0.,cycles=2,
                domains=2,max_batches=2,stop_file=str(stop))
            real_save=c4.Learner.save
            def save_and_stop(obj,path,next_block,config):
                real_save(obj,path,next_block,config)
                if next_block==1:stop.touch()
            with patch.object(c4,'DOMAINS',c4.DOMAINS[:2]),patch.object(robustbench.utils,'load_model',return_value=source),patch.object(c4.Learner,'save',save_and_stop):
                c4.run(args)
            self.assertEqual(len(list(out.glob('block_*/attempt_*/complete.json'))),1)
            stop.unlink()
            with patch.object(c4,'DOMAINS',c4.DOMAINS[:2]),patch.object(robustbench.utils,'load_model',return_value=source):
                c4.run(args)
            self.assertEqual(len(list(out.glob('block_*/attempt_*/complete.json'))),4)
            report.report(out.parent,verify=True)
            self.assertEqual(len(list((out.parent/'report').glob('C4-*.png'))),5)
            import json
            for summary in out.glob('block_*/attempt_*/summary.json'):
                self.assertEqual(json.loads(summary.read_text())['optimizer_steps'],2)


if __name__=='__main__': unittest.main(verbosity=2)
