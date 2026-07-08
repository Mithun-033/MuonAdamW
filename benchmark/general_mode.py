import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from MuonAdamW.optimizer import MuonAdamW
from MuonAdamW.arguments import AdamwArgs
from torch.optim import AdamW as TorchAdamW

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),

            nn.LayerNorm(64),
            nn.Linear(64,128),
            nn.ReLU(),

            nn.LayerNorm(128),
            nn.Linear(128,256),
            nn.ReLU(),

            nn.LayerNorm(256),
            nn.Linear(256,128),
            nn.ReLU(),

            nn.LayerNorm(128),
            nn.Linear(128,64),
            nn.ReLU(),

            nn.LayerNorm(64),
            nn.Linear(64,1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,x):
        return self.model(x)


# Dataset
inputs=torch.randn(50_000,16)

x=inputs

targets=(
    x[:,0]**2
    +torch.sin(3*x[:,1])
    +x[:,2]*x[:,3]
    -torch.exp(-x[:,4]**2)
    +torch.cos(x[:,5]*x[:,6])
    +torch.log1p(x[:,7].abs())
    +torch.tanh(x[:,8]-x[:,9])
    +0.5*x[:,10]**3
    -0.3*x[:,11]*x[:,12]*x[:,13]
    +torch.sin(x[:,14]+x[:,15])
    +0.2*torch.randn(50_000)
).unsqueeze(1)

inputs = (inputs - inputs.mean()) / inputs.std()
targets = (targets - targets.mean()) / targets.std()


loader=DataLoader(
    TensorDataset(inputs,targets),
    batch_size=256,
    shuffle=True,
    pin_memory=True
)

model1=Model().to(device)
model2=Model().to(device)

model1.load_state_dict(model2.state_dict())

learning_rate=1e-9

muonadamw=MuonAdamW(
    model1,
    mode="general",
    lr=learning_rate,
    adam_args=AdamwArgs(weight_decay=0.01)
)

adam=TorchAdamW(model2.parameters(),lr=learning_rate,weight_decay=0.01)

criterion=nn.MSELoss()

muonadamw_loss=[]
adam_loss=[]


with tqdm(loader,desc="Training") as pbar:
    for x,y in pbar:
        x=x.to(device,non_blocking=True)
        y=y.to(device,non_blocking=True)
        # MuonAdamW
        muonadamw.zero_grad()
        pred1=model1(x)
        loss1=criterion(pred1,y)
        loss1.backward()
        muonadamw.step()
        # AdamW
        adam.zero_grad()
        pred2=model2(x)
        loss2=criterion(pred2,y)
        loss2.backward()
        adam.step()
        muonadamw_loss.append(loss1.item())
        adam_loss.append(loss2.item())
        pbar.set_postfix(
            MuonAdamW=f"{loss1.item():.4f}",
            AdamW=f"{loss2.item():.4f}"
        )

plt.figure(figsize=(10,5))
plt.plot(muonadamw_loss,label="MuonAdamW")
plt.plot(adam_loss,label="AdamW")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("loss_comparison.png")