''' 
MuonAdamW is a optimizer package that combines the benefits of AdamW and Muon optimizers. 
It is designed to be compatible with a wide range of deep learning models built using PyTorch, 
including both convolutional neural networks (CNNs) and transformer-based architectures, and a custom mode for user-defined parameter grouping. 
The optimizer allows users to specify which parameters should be optimized with AdamW and which with Muon, 
providing flexibility in how the parameters are optimized.
'''

from . arguments import AdamwArgs, MuonArgs
from . optimizer import MuonAdamW
from . helper_utils import make_adamw, make_muon

