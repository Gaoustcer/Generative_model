import torch.nn as nn
from torchvision import datasets,transforms
from torch.nn.functional import softmax
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('../log/traindiscri')
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
    EPOCH = 4
    testtime = 1
    traintime = 0
    from tqdm import tqdm
    for epoch in (range(EPOCH)):
        for data,label in tqdm(trainloader):
            optimizer.zero_grad()
            data = data.cuda()
            label = label.cuda().to(torch.int64)
            # print("min label",min(label))
            # print("max label",max(label))
            # print('data shape',data.shape)
            result = net(data)
            # # 
            # print(result.shape)
            # print(label.shape)
            # print(label.unsqueeze(-1).shape)
            # exit()
            predict = torch.gather(result,-1,label.unsqueeze(-1)).squeeze(-1)
            # print(predict.shape)
            # print(label.shape)
            loss = lossfunction(predict,torch.ones_like(label).to(torch.float32))
            loss.backward()
            optimizer.step()
            traintime += 1
            # torch.index_select()
            # exit()
            samecount = 0
            totalcount = 0
            if traintime % 32 == 0:
                for data,label in testloader:
                    data = data.cuda()
                    label = label.cuda()
                    netresult = net(data)
                    # print(netresult)
                    testresult = torch.argmax(net(data),-1)
                    samecount += sum(testresult == label)
                    # print(testresult)
                    # print(samecount)
                    # exit()
                    totalcount += len(label)
                writer.add_scalar('acc',samecount/totalcount,testtime)
                testtime += 1

    torch.save(net,'MNIST.pkl')
