import torchvision
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

data_train = datasets.MNIST(root = "../GAN/MNIST_Train/data/",
                            transform=transforms.ToTensor(),
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="../GAN/MNIST_Train/data/",
                           transform = transforms.ToTensor(),
                           train = False)

trainloader = DataLoader(data_train,batch_size=64)
testloader = DataLoader(data_test,batch_size=64)
EPOCH = 8
import torch.nn as nn
from model.vae_net import VAEnet
mselossfunction = nn.MSELoss()
def _train():
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader
    writer = SummaryWriter('./log/vaeloss')
    vae = VAEnet().cuda()

    vaeoptimizer = torch.optim.Adam(vae.parameters(),lr=0.001)
    traintimes = 1
    testtimes = 1
    for epoch in (range(EPOCH)):
        for data,label in tqdm(trainloader):
            data = data.cuda()
            reconstructdata = vae(data)
            vaeoptimizer.zero_grad()
            loss = mselossfunction(reconstructdata,data)
            loss.backward()
            vaeoptimizer.step()
            writer.add_scalar('trainloss',loss,traintimes)
            traintimes += 1
        loss = 0
        for data,label in tqdm(testloader):
            data = data.cuda()
            reconstructdata = vae(data)
            loss = mselossfunction(reconstructdata,data)
            writer.add_scalar('testloss',loss,testtimes)
            testtimes += 1
        torch.save(vae,'vaenetparm{}'.format(str(epoch)))
topilfunction = transforms.ToPILImage()
def _outputpicture(data,path):
    image = topilfunction(data)
    image.save(path)

def _vaildationtrain(path = 'networkparam/vaenetparm7'):
    net = torch.load(path)
    for data,labels in trainloader:
        transdata = net(data)
        for i in range(64):
            origin = data[i]
            trans = transdata[i]
            _outputpicture(origin,'picture/train/origin_{}.png'.format(str(i)))
            _outputpicture(trans,'picture/train/trans_{}.png'.format(str(i)))

        break
if __name__ == "__main__":
    # _train()
    _vaildationtrain()