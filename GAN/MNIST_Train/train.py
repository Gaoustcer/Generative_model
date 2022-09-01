import torch.nn as nn
from torchvision import datasets,transforms
from torch.nn.functional import softmax
from torch.nn import BCELoss
data_train = datasets.MNIST(root = "./data/",
                            transform=transforms.ToTensor(),
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transforms.ToTensor(),
                           train = False)

from torch.utils.data import DataLoader
trainloader = DataLoader(data_train,batch_size=32)
testloader = DataLoader(data_test,batch_size=16)
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,2,3,2),
            nn.ReLU(),
            nn.Conv2d(2,1,3,2),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(36,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,10)
        )
    def forward(self,x):
        return softmax(self.linear(self.net(x)))
import torch

if __name__ == "__main__":
    net = Net().to("cuda:0")
    lossfunction = BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for data,label in trainloader:
        optimizer.zero_grad()
        data = data.cuda()
        label = data.cuda()
        # print("min label",min(label))
        # print("max label",max(label))
        # print('data shape',data.shape)
        result = net(data)
        # # 
        predict = torch.gather(result,-1,result.unsqueeze(-1))
        loss = lossfunction(predict,label)
        optimizer.zero_grad()
        # torch.index_select()
        # exit()
    torch.save(net,'MNIST.pkl')
