from torch import nn
import torch.optim as optim
from typing import Literal,Iterator

class MuonAdamW(optim.Optimizer):
    def __init__(self,
                model : Iterator[nn.Parameter],
                lr : float = 1e-3,
                mode : Literal["shape-wise","role-wise","custom"]="role-wise",
                adamw_betas : tuple[float] = (0.9,0.99),
                adamw_eps : float = 1e-8,
                adamw_weight_decay : float = 0.1,
                adamw_amsgrad : bool = False,
                adamw_maximize : bool = False, 
                adamw_foreach : bool = False,
                adamw_capturable : bool = False,
                adamw_differentiable : bool =False,
                adamw_fused : bool = False,
                muon_weight_decay : float = 1e-2,
                muon_momentum : float = 0.95,
                muon_nesterov : bool = True,
                muon_ns_coefficients : tuple[float] = (3.4445, -4.775, 2.0315),
                muon_eps : float = 1e-7,
                muon_ns_steps : int =5,
                muon_lr_multiplier : str | int = "original",
                ):
        super(MuonAdamW, self).__init__()

        

