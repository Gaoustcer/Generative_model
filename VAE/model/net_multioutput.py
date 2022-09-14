import torch.nn as nn
import torch
from torch.distributions import Normal
class multinet(nn.Module):
    def __init__(self) -> None:
        
        super(multinet,self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU()
        )
        self.mu = nn.Linear(16,4)
        self.sigma = nn.Linear(16,4)
        self.randomgenerate = Normal(0,1)
    
    def forward(self,data):
        data = data.cuda()
        identity = self.seq(data)
        mu,sigma = self.mu(identity),self.sigma(identity)
        noise = self.randomgenerate.sample(mu.shape).cuda()

        return noise * sigma + mu

if __name__ == "__main__":
    Netinst = multinet().cuda()
    vector = torch.randn(64,32)
    ret = Netinst(vector)
    print(ret.shape)
    # mu,sigma = Netinst(vector)
    # print("mu is ",mu.shape)
    # print('sigma is',sigma.shape)