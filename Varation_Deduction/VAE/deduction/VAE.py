import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
class VAE(object):
    def __init__(self,encoder,decoder,traindataset,testdataset) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.trainloader = DataLoader(traindataset,batch_size=64)
        self.testloader = DataLoader(testdataset,batch_size=1)
        self.optimencoder = torch.optim.Adam(self.encoder.parameters(),lr = 0.001)
        self.optimdecoder = torch.optim.Adam(self.decoder.parameters(),lr = 0.001)
        self.transform = T.ToPILImage()   
        self.lossindex = 0 
        self.writer = SummaryWriter("./logs/loss")

    def train(self):
        for images,labels in tqdm(self.trainloader):
            images = images.cuda()
            mu,sigma = self.encoder(images)
            recon_images = self.decoder(sigma,mu)
            different = F.binary_cross_entropy(recon_images.view(-1,28**2),images.view(-1,28**2),reduction='sum')
            # Kldiv = 1 + torch.log(torch.pow(sigma,2)) - torch.pow(sigma,2) - torch.pow(mu,2)
            Kldiv = -0.5 * torch.sum(1 + sigma - torch.pow(mu,2) - torch.exp(sigma))
            # Kldiv = torch.mul(-0.5,Kldiv).mean()
            loss = different + Kldiv
            # loss = different
            self.optimdecoder.zero_grad()
            self.optimencoder.zero_grad()
            loss.backward()
            self.writer.add_scalar('loss',loss,self.lossindex)
            self.lossindex += 1
            self.optimdecoder.step()
            self.optimencoder.step()

    def valid(self,path):
        index = 0
        if os.path.exists(path) == False:
            os.mkdir(path)
        for image,_ in (self.testloader):
            if index == 16:
                return
            image = image.cuda()
            mu,sigma = self.encoder(image)
            picture = self.decoder(mu,sigma).cpu().detach()
            picture = torch.reshape(picture,(28,28))
            image = torch.reshape(image,(28,28)).cpu()
            picture = self.transform(picture)
            image = self.transform(image)
            picture.save(os.path.join(path,"VAE{}.png".format(index)))
            image.save(os.path.join(path,'origin{}.png'.format(index)))

            index+= 1
    
    def deduction(self,path):
        if os.path.exists(path) == False:
            os.mkdir(path)
        for index in range(16):
            mu = torch.zeros((1,16)).to(torch.float32).cuda()
            sigma = torch.ones((1,16)).to(torch.float32).cuda()
            picture = self.decoder(mu,sigma)
            picture = torch.reshape(picture,(28,28)).cpu().detach()
            picture = self.transform(picture)
            picture.save(os.path.join(path,"generate{}.png".format(index)))
        pass

    # def lossfunction(self,sigma,mu)
        