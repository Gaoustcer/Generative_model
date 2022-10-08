from CVAE import CVAE
import torch
from Train_testdataset import data_test,data_train
from torch.utils.data import DataLoader
import os
import torchvision.transforms as T
# loader = DataLoader(data_test)
from torch.utils.tensorboard import SummaryWriter
class Train(object):
    def __init__(self) -> None:
        self.CVAE = CVAE().cuda()
        self.traindata = DataLoader(data_train,batch_size=64)
        self.testdata = DataLoader(data_test,batch_size=1)
        self.EPOCH = 64
        self.writer = SummaryWriter("log/loss")
        self.optimizer = torch.optim.Adam(self.CVAE.parameters(),lr = 0.00001)
    def train(self):
        index = 0
        for epoch in range(self.EPOCH):
            from tqdm import tqdm
            for images,labels in tqdm(self.traindata):
                images = images.cuda()
                labels = labels.cuda()
                labelhot = torch.zeros(images.shape[0],10).cuda()
                labelhot.scatter_(-1,labels.unsqueeze(-1),1)
                picture,mu,sigma = self.CVAE(images,labelhot)
                self.optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(picture,images)
                KLdiv =  torch.log(torch.pow(sigma,2)) - torch.pow(sigma,2) - torch.pow(mu,2)
                # print("KLdivergence",KLdiv.shape)
                # print("Sigma,mu is",sigma.shape,mu.shape)
                KLdiv = -0.5 * KLdiv.sum()
                # print(KLdiv)
                # exit()
                loss += KLdiv
                loss.backward()
                self.writer.add_scalar("loss",loss,index)
                self.optimizer.step()
                index += 1
                if index % 1024 == 0:
                    self.vaildation(index//1024)
    def vaildation(self,index):
        path = "testresult/epoch{}".format(index)
        if "epoch{}".format(index) not in os.listdir("testresult"):   
            os.mkdir(path)
        validindex = 0
        for testimages,labels in self.testdata:
            testimages = testimages.cuda()
            labels = labels.cuda()
            labelhot = torch.zeros(testimages.shape[0],10).cuda()
            labelhot.scatter_(-1,labels.unsqueeze(-1),1)
            constructimages = self.CVAE(testimages,labelhot)[0].squeeze().cpu()
            transform = T.ToPILImage()

            testimages = testimages.squeeze().cpu()
            generateimage = transform(constructimages)
            originimage = transform(testimages)
            generateimage.save(path+"/generate{}.png".format(validindex))
            originimage.save(path+"/origin{}.png".format(validindex))
            validindex += 1
            # print("validindex{}".format(validindex),constructimages)
            # if validindex == 2:
            #     exit()
            if validindex == 16:
                return

if __name__ == "__main__":
    agent = Train()
    agent.train()



                

        