import torch.nn as nn
import torch

from model import Encoder,Decoder
class CVAE(nn.Module):
    def __init__(self) -> None:
        super(CVAE,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self,images,labels):
        assert images.shape[0] == labels.shape[0]
        mu,sigma = self.encoder(images,labels)
        picture = self.decoder(mu,sigma,labels)
        return torch.reshape(picture,[mu.shape[0],28,28])
