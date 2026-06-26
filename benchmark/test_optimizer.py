import unittest
from torch import nn
import torch

from MuonAdamW.optimizer import MuonAdamW
from MuonAdamW.arguments import AdamW, Muon

class TestMuonAdamW(unittest.TestCase):
    def test_optimizer_initialization(self):
        
        
        # Test with default parameters
        model = nn.Linear(10, 1)
        optimizer = MuonAdamW(model)
        self.assertIsInstance(optimizer, MuonAdamW)
        
        # Test with custom parameters
        adam_args = AdamW(betas=(0.8, 0.88), eps=1e-6)
        muon_args = Muon(weight_decay=0.05, momentum=0.9)
        optimizer = MuonAdamW(model, mode="transformer", muon_lr_multiplier="match_rms_adamw", adam_args=adam_args, muon_args=muon_args)
        self.assertIsInstance(optimizer, MuonAdamW)

    def test_invalid_mode(self):
        model = nn.Linear(10, 1)
        with self.assertRaises(AssertionError):
            MuonAdamW(model, mode="invalid_mode")
        
    def test_invalid_muon_lr_multiplier(self):
        model = nn.Linear(10, 1)
        with self.assertRaises(AssertionError):
            MuonAdamW(model, muon_lr_multiplier="invalid_multiplier")

    def test_custom_mode_without_parameters(self):
        model = nn.Linear(10, 1)
        with self.assertRaises(ValueError):
            MuonAdamW(model, mode="custom")

    def test_custom_mode_with_parameters(self):
        model = nn.Linear(10, 1)
        muon_params = [param for param in model.parameters() if param.dim() > 1]
        adam_params = [param for param in model.parameters() if param.dim() == 1]
        optimizer = MuonAdamW(model, mode="custom", muon_parameters=muon_params, adam_parameters=adam_params)
        self.assertIsInstance(optimizer, MuonAdamW)

    def test_parameter_grouping_transformer(self):
        model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
        optimizer = MuonAdamW(model, mode="transformer")
        self.assertIsInstance(optimizer, MuonAdamW)
        self.assertGreater(len(optimizer.adamw_params), 0)
        self.assertGreater(len(optimizer.muon_params), 0)

    def test_scheduler_integration(self):
        model = nn.Linear(10, 1)
        optimizer = MuonAdamW(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


    def test_scheduler_step(self):
        model = nn.Linear(10, 1)
        optimizer = MuonAdamW(model, muon_lr_multiplier="match_rms_adamw")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        initial_adamw_lr = optimizer.param_groups[0]['  lr']
        initial_muon_lr = optimizer.param_groups[1]['lr']   

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        updated_adamw_lr = optimizer.param_groups[0]['lr']
        updated_muon_lr = optimizer.param_groups[1]['lr']

        self.assertNotEqual(initial_adamw_lr, updated_adamw_lr)
        self.assertNotEqual(initial_muon_lr, updated_muon_lr)

        print(f"Initial AdamW LR: {initial_adamw_lr}, Updated AdamW LR: {updated_adamw_lr}")
        print(f"Initial (pre-scaled) Muon LR: {initial_muon_lr}, Updated (pre-scaled) Muon LR: {updated_muon_lr}")

if __name__ == '__main__':
    unittest.main(verbosity=2)