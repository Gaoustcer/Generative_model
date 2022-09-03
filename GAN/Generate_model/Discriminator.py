import torch.nn as nn
from torch.nn.functional import softmax
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator,self).__init__()
        self.indentifier = nn.Sequential(
            nn.Linear(1024,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    
    def forward(self,x):
        return softmax(self.indentifier(x))