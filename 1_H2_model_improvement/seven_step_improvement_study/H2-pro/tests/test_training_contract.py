import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1]/'implementation'/'software')]
import run_stage1 as stage


class TrainingContractTests(unittest.TestCase):
    def test_reserved_roles_rejected(self):
        for role in ('validation','test','hardware_calibration'):
            with self.assertRaises(ValueError):
                list(stage._role_files(role,1))

    def test_frozen_hashes(self):
        self.assertEqual(len(stage.verify_inputs()),3)

    def test_float_checkpoint_initial_column_zero(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is not installed in this interpreter')
        model = stage._model_from_checkpoint(stage.SOURCE_FLOAT)
        first = model.combined()[0].detach().numpy()
        self.assertEqual(first.shape,(64,36))
        self.assertTrue(np.all(first[:,2] == 0))
        layers = stage._quantized(model)
        self.assertEqual(layers[0][0].dtype,np.int8)
        self.assertTrue(np.all(layers[0][0][2] == 0))

    def test_warmup_changes_only_new_column(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest('PyTorch is not installed in this interpreter')
        model = stage._model_from_checkpoint(stage.SOURCE_FLOAT)
        for module in (model.old,model.hidden,model.output):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        before = [weight.detach().clone() for weight in model.combined()]
        optimizer = torch.optim.AdamW(model.new.parameters(),lr=0.001,weight_decay=0.0001)
        values = torch.randint(0,256,(32,36),dtype=torch.uint8)
        values[:,2] = 128
        truth = torch.randint(0,2,(32,),dtype=torch.float32)
        logits = model(values).reshape(-1,2,2).amax(2)
        loss = F.binary_cross_entropy(torch.softmax(logits,dim=1)[:,1],truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = model.combined()
        self.assertTrue(torch.equal(before[0][:,:2],after[0][:,:2]))
        self.assertTrue(torch.equal(before[0][:,3:],after[0][:,3:]))
        self.assertFalse(torch.equal(before[0][:,2],after[0][:,2]))
        self.assertTrue(torch.equal(before[1],after[1]))
        self.assertTrue(torch.equal(before[2],after[2]))


if __name__ == '__main__':
    unittest.main()
