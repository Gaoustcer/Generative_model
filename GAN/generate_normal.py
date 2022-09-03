# from  .Discriminator import Discriminator
# from .Generator import Generator
from Generate_model.Discriminator import Discriminator
from Generate_model.Generator import Generator
# from Generate_model.Generator import Generator
# from .Generator import Generator
from torch.distributions import Normal
Distributionsize = 1024
BATCHSIZE = 32
EPOCH = 1024
UPDATEDTIME = 16
import torch
import matplotlib.pyplot as plt
D_net = Discriminator().to(torch.device('cuda:0'))
D_optim = torch.optim.Adam(D_net.parameters(),lr=0.00001)
G_net = Generator().to(torch.device('cuda:0'))
G_optimizer = torch.optim.Adam(G_net.parameters(),lr=0.00001)
from torch.nn import BCELoss

# def print(x,y,c,figname):
#     plt.scatter(x,y,c=c)
#     plt.savefig(figname)
#     plt.close()
def _notrain():
    x = torch.rand(32).cuda()
    result = G_net(x).detach().cpu()
    plt.hist(result,bins=64)
    plt.savefig('./pic/train_from_scratch.png')
    plt.close()

if __name__ == "__main__":
    _notrain()
    from tqdm import tqdm
    normal = Normal(0,1)
    lossfunc = BCELoss()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./log/GANtrain')
    for epoch in tqdm(range(EPOCH)):
        for _ in range(UPDATEDTIME):
            realsample = normal.sample((BATCHSIZE,Distributionsize)).cuda()
            fakesample = torch.rand((BATCHSIZE,32)).cuda()
            fakeresult = G_net(fakesample).detach()
            D_optim.zero_grad()
            loss = 0
            realpredict = D_net(realsample)
            # print("shape of predict is ",realpredict.shape)
            # print("possibility is",realpredict)
            # print("mean is ",realpredict.mean())
            # exit()
            loss -= realpredict.mean()
            # -log p_real p_real is larger
            fakepredict = D_net(fakeresult)
            fakepredict = torch.ones_like(fakepredict) - fakepredict
            # print("fake is",fakepredict)
            # exit()
            loss -= fakepredict.mean()
            # -log (1-p_real)=-log p_fake
            # loss = log p_real + log p_fake
            loss.backward()
        G_optimizer.zero_grad()
        noise = torch.rand((BATCHSIZE,32)).cuda()
        GeneratorDiscrim = D_net(G_net(noise))
        # loss = 0
        GeneratorDiscrim = torch.ones_like(GeneratorDiscrim) - GeneratorDiscrim
        loss = GeneratorDiscrim.mean()
        # loss = -log p_real
        # likelihood is p,BCEL return -log p for Generator we want p is samaller 
        loss.backward()
        writer.add_scalar('lossgan',loss,epoch)
        G_optimizer.step()
        if epoch % 128 == 0:
            # noise = torch.rand((1,32)).cuda()
            noise = noise[0]
            noise = torch.reshape(noise,(1,32))
            # print("noise shape",noise.shape)
            dis = G_net(noise).squeeze(0).detach().cpu()
            plt.hist(dis,bins=64)
            plt.savefig("./pic/figure"+str(int(epoch/128))+'.png')
            print("from real data",D_net(G_net(noise)))
            plt.close()
            # print(dis.shape)
            # exit()

    # import numpy as np
    # c = ['r','g']
    # figname = ['fig1.png','fig2.png']
    # for i in range(2):
    #     x = np.random.random(32)
    #     y = np.random.random(32)
    #     print(x,y,c[i],figname[i])
