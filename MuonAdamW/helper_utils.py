import torch.optim as optim
from arguments import AdamW, Muon

def make_adamw(lr : float, adam_parameters : list, config : AdamW):
    '''Helper function to create an AdamW optimizer with the specified parameters and configuration.
    Args:
        lr (float): Learning rate for the AdamW optimizer.
        adam_parameters (list): List of parameters to be optimized by AdamW.
        config (AdamW): Configuration object containing AdamW hyperparameters.
    Returns:
        torch.optim.AdamW: An instance of the AdamW optimizer initialized with the provided parameters and configuration.
    '''
    optimizer = optim.AdamW(
        adam_parameters,
        lr=lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
        maximize=config.maximize,
        foreach=config.foreach,
        capturable=config.capturable,
        differentiable=config.differentiable,
        fused=config.fused
    )

    return optimizer

def make_muon(lr : float, muon_parameters : list, muon_lr_multiplier : float | str, config : Muon):
    '''Helper function to create a Muon optimizer with the specified parameters and configuration.
    Args:
        lr (float): Base learning rate for the Muon optimizer.
        muon_parameters (list): List of parameters to be optimized by Muon.
        muon_lr_multiplier (float | str): Multiplier for the learning rate of the Muon optimizer. 
            Can be a float value or a string indicating whether to use the original learning rate or to match the RMS of AdamW updates.
        config (Muon): Configuration object containing Muon hyperparameters.
    Returns:
        torch.optim.Muon: An instance of the Muon optimizer initialized with the provided parameters and configuration.
    '''
    if muon_lr_multiplier == "original" or muon_lr_multiplier == "match_rms_adamw":
        muon_lr = lr

    if isinstance(muon_lr_multiplier, float):
        muon_lr = lr * muon_lr_multiplier

    optimizer = optim.Muon(
        muon_parameters,
        lr=muon_lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_coefficients=config.ns_coefficients,
        eps=config.eps,
        ns_steps=config.ns_steps,
        adjust_lr_fn = muon_lr_multiplier if muon_lr_multiplier in ["original","match_rms_adamw"] else None
    )

    return optimizer