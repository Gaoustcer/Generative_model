import torch.nn as nn
import torch
device = torch.device('cuda:0')
class net(nn.Module):
    def __init__(self) -> None:
        super(net,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,1)
        ).to(device)
    def forward(self,x):
        return self.layer(x)
from matplotlib import pyplot as plt


def train():
    EPOCH = 1024
    BATCHSIZE = 1024
    lossfunc = nn.MSELoss()
    Net = net()
    optimizer = torch.optim.Adam(Net.parameters(),lr = 0.001)
    from tqdm import tqdm
    for epoch in tqdm(range(EPOCH)):
        optimizer.zero_grad()
        # sampledata = []
        tensor = torch.rand((BATCHSIZE,1)).to(device)
        result = Net(tensor)
        randomresult = torch.normal(0,1,(BATCHSIZE,1)).to(device)
        loss = lossfunc(result,randomresult)
        loss.backward()
        optimizer.step()
    tensor = torch.rand((BATCHSIZE ** 2,1)).to(device)
    result = Net(tensor)
    print("Net forward finished")
    result = result.squeeze(-1).cpu().detach()
    groundtruth = torch.normal(0,1,(BATCHSIZE**2,1))
    print("generate groundtruth")
    result = result.numpy()
    groundtruth = groundtruth.squeeze(-1).numpy()
    print("generate numpy")
    # plt.hist(result,bins=BATCHSIZE)
    plt.hist(groundtruth,bins=BATCHSIZE)
    plt.savefig("groundtruth.png")
        # result = result.squeeze(-1).cpu().detach()
        # plt.hist(result,bins=1024)
        # plt.savefig("./picture/{}.png".format(str(epoch)))



if __name__ == "__main__":
    train()
    # a = torch.rand((32,1)).to(device)
    # Net = net()
    # print(Net(a))
