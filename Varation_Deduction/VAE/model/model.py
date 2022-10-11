import torch.nn as nn
import torch
class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder,self).__init__()
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28**2,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.muencode = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )
        from copy import deepcopy
        self.sigmaencode = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16)
            # nn.ReLU()
        )
    
    def forward(self,images):
        feature = self.feature(images)
        return self.muencode(feature),self.sigmaencode(feature)

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()
        self.Decode = nn.Sequential(
            nn.Linear(16,128),
            nn.ReLU(),
            nn.Linear(128,28**2),
            nn.Sigmoid()
        )
    
    def forward(self,sigma,mu):
        noise = torch.randn_like(sigma).cuda()
        sigma = torch.exp(0.5 * sigma)
        # output result is log sigma
        feature = noise * sigma + mu
        return self.Decode(feature)


if __name__ == "__main__":
    net = Encoder(
    )
    net.parameters