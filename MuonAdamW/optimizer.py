'''
This module implements the MuonAdamW optimizer, which combines the AdamW and Muon optimization algorithms.
The MuonAdamW optimizer allows for flexible configuration of both AdamW and Muon components,
enabling users to tailor the optimization process to their specific needs. The optimizer supports three modes of operation:
1. Shape-wise: Parameters are categorized based on their shape, with multi-dimensional parameters optimized using
   Muon and one-dimensional parameters optimized using AdamW.
2. Role-wise: Parameters are categorized based on their role in the model. Embeddings, Norms are optimized using AdamW, 
    while all other 2d parameters are optimized using Muon.
3. Custom: Users can explicitly specify which parameters should be optimized with Muon and which with AdamW.
The optimizer also allows for different learning rates for the Muon and AdamW components, with options
to maintain the original learning rate or to match the RMS of AdamW updates.

'''

from torch import nn
import torch.optim as optim
from typing import Literal,Iterator
from arguments import AdamW,Muon
from helper_utils import make_adamw, make_muon

class MuonAdamW(optim.Optimizer):
    '''Optimizer that combines AdamW and Muon optimizers.'''
    def __init__(self,
                model : Iterator[nn.Parameter],
                lr : float = 1e-3,
                mode : Literal["cnn","transformer","custom"]="role-wise",
                muon_parameters : list[nn.Parameter] | None = None,
                adam_parameters : list[nn.Parameter] | None = None,
                adam_args : AdamW | None = None,
                muon_args : Muon | None = None,
                muon_lr_multiplier : Literal["original","match_rms_adamw"] | float = "original",
                ):
    
        super(MuonAdamW, self).__init__()
        
        assert mode in ["cnn","transformer","custom"], "Invalid mode. Supported modes are 'cnn', 'transformer', and 'custom'."
        assert muon_lr_multiplier == "original" or muon_lr_multiplier == "match_rms_adamw" or isinstance(muon_lr_multiplier, float), "Invalid muon_lr_multiplier. Supported values are 'original', 'match_rms_adamw', or a float value."
        if adam_args is None:
            adam_args = AdamW()
        if muon_args is None:
            muon_args = Muon()

        self.adam_linear=set("lm_head","final_head","classifier","head")
        self.mode = mode
        self.muon_lr_multiplier = muon_lr_multiplier  

        if self.mode == "custom":
            if muon_parameters is None or adam_parameters is None:
                raise ValueError("For custom mode, both muon_parameters and adam_parameters must be provided in a list format.")
            self.adam_params = adam_parameters
            self.muon_params = muon_parameters
        
        elif self.mode == "transformer":
            self.muon_params = []
            self.adam_params = []
            for module in model.modules():
                if isinstance(module, (nn.Embedding, nn.LayerNorm)):
                    self.adam_params.extend(list(module.parameters()))
                else:
                    for name, param in module.named_parameters():
                        # A heuristic to identify final projection layers in transformers, 
                        # which are typically linear layers. This is not a foolproof method and 
                        # may need adjustments based on the specific model architecture.
                        if name in self.adam_linear: 
                            self.adam_params.append(param)
                            continue
                        if param.dim() > 1:
                            self.muon_params.append(param)
                        else:
                            self.adam_params.append(param)

        elif self.mode == "cnn":
            self.muon_params = []
            self.adam_params = []
            flag = True
            for module in model.modules():
                if flag and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    self.muon_params.extend(list(module.parameters()))
                    flag = False
                    continue

                for name, param in module.named_parameters():
                    if name in self.adam_linear: 
                        self.adam_params.append(param)
                        continue
                    if param.dim() > 1:
                        self.muon_params.append(param)
                    else:
                        self.adam_params.append(param)

        self.adamw= make_adamw(lr, self.adam_params, adam_args)
        self.muon = make_muon(lr, self.muon_params, muon_lr_multiplier, muon_args)

    def step(self, closure=None):
        self.adamw.step(closure)
        self.muon.step(closure)

    def zero_grad(self):
        self.adamw.zero_grad()
        self.muon.zero_grad()

    def state_dict(self):
        return {
            'adamw': self.adamw.state_dict(),
            'muon': self.muon.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.adamw.load_state_dict(state_dict['adamw'])
        self.muon.load_state_dict(state_dict['muon'])

    def add_param_group(self, adam_param_group=None, muon_param_group=None):
        if adam_param_group is not None:
            self.adamw.add_param_group(adam_param_group)
        if muon_param_group is not None:
            self.muon.add_param_group(muon_param_group)

    


        
        


        

