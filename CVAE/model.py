import torch.nn as nn
from copy import deepcopy
from Train_testdataset import data_train
class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder,self).__init__()
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28**2,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,8)
        )
        self.sigma = nn.Sequential(
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,4)
        )
        self.labelfeature = nn.Sequential(
            nn.Linear(10,16),
            nn.ReLU(),
            nn.Linear(16,8)
        )
        self.mu = deepcopy(self.sigma)
    
    def forward(self,images,labels):
        features = self.feature(images)
        labelfeature = self.labelfeature(labels)
        # print(labelfeature.shape)
        # print("Feature shape is",labelfeature.shape)
        features = torch.concat([features,labelfeature],-1).to(torch.float32)
        # print("concat shape is",features.shape)
        return self.mu(features),self.sigma(features)
import torch
from torch.distributions import Normal
class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()
        self.noisegenerate = Normal(0,1)
        self.labelEncoder = nn.Sequential(
            nn.Linear(10,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4)
        )
        self.Picture = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,28**2),
            nn.Sigmoid()
        )
    def forward(self,mu,sigma,labels):
        noise = self.noisegenerate.sample(mu.shape).cuda() * sigma + mu
        labelfeature = self.labelEncoder(labels)
        features = torch.concat([noise,labelfeature],-1)
        # print("The Feature shape is",features.shape)
        return self.Picture(features)
    # pass


if __name__ == "__main__":
    net = Encoder()
    decoder = Decoder()
    from torch.utils.data import DataLoader
    loader = DataLoader(data_train,batch_size=64)
    for images,labels in loader:
        # sigma,mu = net(images)
        labelonhot = torch.zeros(images.shape[0],10)
        labelonhot.scatter_(-1,labels.unsqueeze(-1),1)
        print("labels is",labels.shape)
        mu,sigma = net(images,labelonhot)
        print(sigma.shape)
        print(mu.shape)
        pic = decoder(mu,sigma,labelonhot)
        print("Picture shape is",pic.shape)
        exit()
