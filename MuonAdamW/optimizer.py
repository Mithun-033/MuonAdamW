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
from typing import Literal
from torch import nn
import torch.optim as optim
from arguments import AdamW,Muon
from helper_utils import make_adamw, make_muon

class MuonAdamW(optim.Optimizer):
    '''Optimizer that combines AdamW and Muon optimizers.'''
    def __init__(self,
                model : nn.Module,
                lr : float = 1e-3,
                mode : Literal["cnn","transformer","custom"]="transformer",
                muon_parameters : list[nn.Parameter] | None = None,
                adam_parameters : list[nn.Parameter] | None = None,
                adam_args : AdamW | None = None,
                muon_args : Muon | None = None,
                muon_lr_multiplier : Literal["original","match_rms_adamw"] | float = "original",
                ):
        
        assert mode in ["cnn","transformer","custom"], "Invalid mode. Supported modes are 'cnn', 'transformer', and 'custom'."
        assert muon_lr_multiplier == "original" or muon_lr_multiplier == "match_rms_adamw" or isinstance(muon_lr_multiplier, float), "Invalid muon_lr_multiplier. Supported values are 'original', 'match_rms_adamw', or a float value."
        
        if adam_args is None:
            adam_args = AdamW()
        if muon_args is None:
            muon_args = Muon()

        self.adamw_linear=set(["lm_head","final_head","classifier","head"])
        self.mode = mode
        self.muon_lr_multiplier = muon_lr_multiplier  

        self.muon_params = []
        self.adamw_params = []
        
        if self.mode == "custom":
            if muon_parameters is None or adam_parameters is None:
                raise ValueError("For custom mode, both muon_parameters and adam_parameters must be provided in a list format.")
            self.muon_params = muon_parameters
            self.adamw_params = adam_parameters
        
        elif self.mode == "transformer":
            for module in model.modules():
                if isinstance(module, (nn.Embedding, nn.LayerNorm)):
                    self.adamw_params.extend(list(module.parameters()))
                else:
                    for name, param in module.named_parameters():
                        # A heuristic to identify final projection layers in transformers, 
                        # which are typically linear layers. This is not a foolproof method and 
                        # may need adjustments based on the specific model architecture.
                        if name in self.adamw_linear: 
                            self.adamw_params.append(param)
                            continue
                        if param.dim() > 1:
                            self.muon_params.append(param)
                        else:
                            self.adamw_params.append(param)

        elif self.mode == "cnn":
            flag = True
            for module in model.modules():
                if flag and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    self.adamw_params.extend(list(module.parameters()))
                    flag = False
                    continue

                for name, param in module.named_parameters():
                    if name in self.adamw_linear: 
                        self.adamw_params.append(param)
                        continue
                    if param.dim() > 1:
                        self.muon_params.append(param)
                    else:
                        self.adamw_params.append(param)

        param_groups = [
            {
                "params": self.adamw_params,
                "optimizer": "adamw"
            },
            {
                "params": self.muon_params,
                "optimizer": "muon"
            }
        ]
        defaults = {
            "adamw": adam_args,
            "muon": muon_args
        }

        self.adamw= make_adamw(lr, self.adamw_params, adam_args)
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
        self.adamw.load_state_dict(state_dict['adamw'])
        self.muon.load_state_dict(state_dict['muon'])

    
MuonAdamW.__doc__ = r'''
Implementation of the MuonAdamW optimizer, which combines AdamW and Muon.

MuonAdamW automatically splits parameters between AdamW and Muon
optimizers based on the selected mode:

- "transformer": Embeddings, normalization layers, and output heads
  use AdamW, while other multi-dimensional parameters use Muon.
- "cnn": The first convolution layer uses AdamW, while other
  multi-dimensional parameters use Muon.
- "custom": Parameters are explicitly assigned by the user.

The optimizer supports independent hyperparameter configurations
for both AdamW and Muon, as well as optional Muon learning-rate
scaling.

Args:
    model (nn.Module):
        Model whose parameters will be optimized.

    lr (float, optional):
        Base learning rate. Default: 1e-3.

    mode (Literal["cnn", "transformer", "custom"], optional):
        Parameter partitioning strategy. Default: "transformer".

    muon_parameters (list[nn.Parameter] | None, optional):
        Parameters assigned to Muon when using custom mode.

    adam_parameters (list[nn.Parameter] | None, optional):
        Parameters assigned to AdamW when using custom mode.

    adam_args (AdamW | None, optional):
        AdamW configuration dataclass. If None, default values
        from AdamW are used.

    muon_args (Muon | None, optional):
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