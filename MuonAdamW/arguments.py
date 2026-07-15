from dataclasses import dataclass 

@dataclass
class MuonArgs:
    '''
    Arguments for the Muon optimizer.
    Mostly consistent with PyTorch's default
    '''
    weight_decay : float = 1e-2
    momentum : float = 0.95
    nesterov : bool = True
    ns_coefficients : tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    eps : float = 1e-7
    ns_steps : int = 5

@dataclass
class AdamwArgs:
    '''
    Arguments for the AdamW optimizer.
    Mostly consistent with PyTorch's default
    '''
    betas : tuple[float, float] = (0.9,0.99)
    eps : float = 1e-8
    weight_decay : float = 0.1
    amsgrad : bool = False
    maximize : bool = False
    foreach : bool = False
    capturable : bool = False
    differentiable : bool =False
    fused : bool = False