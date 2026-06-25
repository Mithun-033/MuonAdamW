import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from MuonAdamW.optimizer import MuonAdamW
from MuonAdamW.arguments import AdamW
from torch.optim import AdamW as TorchAdamW

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(1,64),
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

    def forward(self,x):
        return self.model(x)


# Dataset
inputs=(torch.randn(50_000,1)).float()
targets=(
    inputs**3
    -inputs**2
    +inputs*1.5
    -inputs*0.5
    -torch.cos(inputs)**2
    +torch.sin(inputs)**2
    +torch.randn(50_000,1)*0.1
) 

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

learning_rate=1e-9

muonadamw=MuonAdamW(
    model1,
    mode="general",
    lr=learning_rate,
    adam_args=AdamW(weight_decay=0.01)
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