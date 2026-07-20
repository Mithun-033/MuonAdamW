'''
This module implements the MuonAdamW optimizer, which combines the AdamW and Muon optimizers. 
It provides a flexible interface for parameter partitioning and supports independent hyperparameter 
configurations for both optimizers.
Modes :-
    general :-
        In this mode, all multi-dimensional parameters (e.g., weights of linear layers) are optimized using Muon,
        while all other parameters (e.g., biases, LayerNorm weights) are optimized using AdamW.
    transformer :-
        In this mode, embeddings, normalization layers, and output heads (if weights are tied) are optimized using AdamW,
        while all other multi-dimensional parameters are optimized using Muon.
    custom :-
        In this mode, the user explicitly specifies which parameters are optimized using Muon and which are optimized using AdamW.
        The user must provide two lists of parameters: one for Muon and one for AdamW.

'''
from typing import Iterator, Literal
from torch import nn
import torch.optim as optim
from .arguments import AdamwArgs,MuonArgs
from .helper_utils import make_adamw, make_muon

class MuonAdamW(optim.Optimizer):
    '''Optimizer that combines AdamW and Muon optimizers.'''
    def __init__(self,
                model : nn.Module,
                lr : float = 1e-3,
                mode : Literal["general","transformer","custom"]="general",
                muon_parameters : list[nn.Parameter] | Iterator |None = None,
                adamw_parameters : list[nn.Parameter] | Iterator |None = None,
                adamw_args : AdamwArgs | None = None,
                muon_args : MuonArgs | None = None,
                muon_lr_multiplier : Literal["original","match_rms_adamw"] | float = "original",
                ):
        
        assert mode in ["general","transformer","custom"], "Invalid mode. Supported modes are 'general', 'transformer', and 'custom'."
        assert muon_lr_multiplier == "original" or muon_lr_multiplier == "match_rms_adamw" or isinstance(muon_lr_multiplier, float), "Invalid muon_lr_multiplier. Supported values are 'original', 'match_rms_adamw', or a float value."
        assert isinstance(model, nn.Module), "The model must be an instance of torch.nn.Module."
        
        if adamw_args is None:
            adamw_args = AdamwArgs()
        if muon_args is None:
            muon_args = MuonArgs()

        self.mode = mode
        self.muon_lr_multiplier = muon_lr_multiplier  

        self.muon_params = []
        self.adamw_params = []
        
        if self.mode == "custom":
            if muon_parameters is None or adamw_parameters is None:
                raise ValueError("For custom mode, both muon_parameters and adamw_parameters must be provided in a list format.")
            self.muon_params = list(muon_parameters)
            self.adamw_params = list(adamw_parameters)

        elif self.mode == "general":
            for param in model.parameters():
                if param.dim() > 1:
                    self.muon_params.append(param)
                else:
                    self.adamw_params.append(param)
        
        elif self.mode == "transformer":
            seen = set()
            for module in model.modules():
                if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    for param in module.parameters():
                        if id(param) not in seen:
                            self.adamw_params.append(param)
                            seen.add(id(param))
                
                for param in module.parameters(recurse = False):
                    if id(param) not in seen:
                        if param.dim() > 1: 
                            self.muon_params.append(param)
                        else:
                            self.adamw_params.append(param)
                        seen.add(id(param))
                

        param_groups = [
            {
                "params": self.adamw_params,
                "optimizer": "adamw",
                "lr" : lr
            },
            {
                "params": self.muon_params,
                "optimizer": "muon",
                "lr" : lr * (muon_lr_multiplier if isinstance(muon_lr_multiplier, float) else 1.0)
            }
        ]
        defaults = {
            "adamw": adamw_args,
            "muon": muon_args
        }

        self.adamw= make_adamw(lr, self.adamw_params, adamw_args)
        self.muon = make_muon(lr, self.muon_params, muon_lr_multiplier, muon_args)

        super().__init__(param_groups, defaults)

    def step(self, closure=None):
        self.adamw.step(closure)
        self.muon.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.adamw.zero_grad(set_to_none)
        self.muon.zero_grad(set_to_none)

    def state_dict(self):
        return {
            'adamw': self.adamw.state_dict(),
            'muon': self.muon.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        assert 'adamw' in state_dict and 'muon' in state_dict, "State dict must contain both 'adamw' and 'muon' keys."
        self.adamw.load_state_dict(state_dict['adamw'])
        self.muon.load_state_dict(state_dict['muon'])

    def add_param_group(self, param_group):
        assert "muon" in param_group and "adamw" in param_group, "Param group must contain both 'muon' and 'adamw' keys."
        self.muon.add_param_group(param_group["muon"])
        self.adamw.add_param_group(param_group["adamw"])

MuonAdamW.__doc__ = r'''
Implementation of the MuonAdamW optimizer, which combines AdamW and Muon.

MuonAdamW automatically splits parameters between AdamW and Muon
optimizers based on the selected mode:

- "general": All multi-dimensional parameters use Muon, while other parameters use AdamW.
- "transformer": Embeddings, normalization layers, and output heads
  use AdamW, while other multi-dimensional parameters use Muon.
- "custom": Parameters are explicitly assigned by the user.

The optimizer supports independent hyperparameter configurations
for both AdamW and Muon, as well as optional Muon learning-rate
scaling.

Args:
    model (nn.Module):
        Model whose parameters will be optimized.

    lr (float, optional):
        Base learning rate. Default: 1e-3.

    mode (Literal["general", "transformer", "custom"], optional):
        Parameter partitioning strategy. Default: "transformer".

    muon_parameters (list[nn.Parameter] | None, optional):
        Parameters assigned to Muon when using custom mode.

    adamw_parameters (list[nn.Parameter] | None, optional):
        Parameters assigned to AdamW when using custom mode.

    adamw_args (AdamWArgs | None, optional):
        AdamW configuration dataclass. If None, default values
        from AdamW are used.

    muon_args (MuonArgs | None, optional):
        Muon configuration dataclass. If None, default values
        from Muon are used.

    muon_lr_multiplier (Literal["original", "match_rms_adamw"] | float, optional):
        Controls Muon's effective learning rate.

        - "original": Use the base learning rate.
        - "match_rms_adamw": Match AdamW update RMS.
        - float: Scale Muon learning rate by the given factor.

Raises:
    AssertionError:
        If mode or muon_lr_multiplier is invalid.

    ValueError:
        If custom mode is selected without providing both
        muon_parameters and adam_parameters.
'''
        
if __name__ == "__main__":
    print(MuonAdamW.__doc__)