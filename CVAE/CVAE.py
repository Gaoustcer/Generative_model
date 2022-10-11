import torch.nn as nn
import torch

from model import Encoder,Decoder
from torch.distributions import Normal
class CVAE(nn.Module):
    def __init__(self) -> None:
        super(CVAE,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.latent_dim = 4
        self.randomsample = Normal(0,1)
    
    def forward(self,images,labels):
        assert images.shape[0] == labels.shape[0]
        mu,sigma = self.encoder(images,labels)
        picture = self.decoder(mu,sigma,labels)
        return torch.reshape(picture,[mu.shape[0],28,28]),mu,sigma
    
    def decode(self,labels):
        # noise = self.randomsample.sample((labels.shape[0],self.latent_dim))
        return self.decoder.decode(labels)
        # return self.decoder()
