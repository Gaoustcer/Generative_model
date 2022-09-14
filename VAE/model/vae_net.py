import torch.nn as nn
import torch
from torch.distributions import Normal
Net_parameter = [28 ** 2,512,256,128,64,32]

class VAEnet(nn.Module):
    def __init__(self) -> None:
        super(VAEnet,self).__init__()
        self.normalgenerate = Normal(0,1)
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            # nn.Linear(28**2,128),
            # nn.ReLU(),
            # nn.Linear(128,64),
            # nn.ReLU()
        )
        for i in range(len(Net_parameter)-1):
            self.encoder.add_module('encoder {}'.format(str(i)),nn.Linear(Net_parameter[i],Net_parameter[i+1]))
            self.encoder.add_module('relu {}'.format(str(i)),nn.ReLU())
        self.decoder = nn.Sequential()
        self.sigma = nn.Linear(32,32)
        self.mu = nn.Linear(32,32)

        for i in range(-1,-len((Net_parameter)),-1):
            self.decoder.add_module('decoder {}'.format(str(i)),nn.Linear(Net_parameter[i],Net_parameter[i-1]))
            self.decoder.add_module('relu {}'.format(str(i)),nn.ReLU())
        
    def forward(self,data:torch.Tensor):
        shape_x = data.shape[0]
        data = data.cuda()
        data = self.flatten(data)
        identity = self.encoder(data)
        sigma = self.sigma(identity)
        mu = self.mu(identity)
        noise = self.normalgenerate.sample(identity.shape).cuda()
        decoderinput = noise * sigma + mu
        
        return torch.reshape(self.decoder(decoderinput),(shape_x,1,28,28))


if __name__ == "__main__":
    data = torch.randn((32,1,28,28))
    net = VAEnet().cuda()
    ret = net(data)
    print(ret.shape)

    