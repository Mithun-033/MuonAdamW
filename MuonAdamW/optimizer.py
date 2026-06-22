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
                mode : Literal["shape-wise","role-wise","custom"]="role-wise",
                muon_parameters : list[nn.Parameter] | None = None,
                adam_parameters : list[nn.Parameter] | None = None,
                adam_args : AdamW | None = None,
                muon_args : Muon | None = None,
                muon_lr_multiplier : Literal["original","match_rms_adamw"] | float = "original",
                ):
    
        super(MuonAdamW, self).__init__()
        
        if adam_args is None:
            adam_args = AdamW()
        if muon_args is None:
            muon_args = Muon()

        self.mode = mode
        self.muon_lr_multiplier = muon_lr_multiplier  

        if self.mode == "custom":
            assert muon_parameters is not None and adam_parameters is not None, '''For custom mode, both muon_parameters and adam_parameters must be provided.'''
            self.muon_params = muon_parameters
            self.adam_params = adam_parameters

        elif self.mode == "shape-wise":
            self.muon_params = []
            self.adam_params = []
            for param in model:
                if param.ndim >= 2:
                    self.muon_params.append(param)
                else:
                    self.adam_params.append(param)  

        elif self.mode == "role-wise":
            ...

        self.adamw_optimizer = make_adamw(lr, self.adam_params, adam_args)
        self.muon_optimizer = make_muon(lr, self.muon_params, muon_lr_multiplier, muon_args)
        
        
        


        

