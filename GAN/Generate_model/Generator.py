from torch.nn import Module
import torch.nn as nn
import torch
BATCH_SIZE = 1024
class Generator(Module):
    def __init__(self) -> None:
        super(Generator,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,BATCH_SIZE)
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.zeros_(m.weight.data)
    
    def forward(self,x:torch.tensor):
        return self.linear(x)
