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
D_optim = torch.optim.Adam(D_net.parameters(),lr=0.001)
G_net = Generator().to(torch.device('cuda:0'))
G_optimizer = torch.optim.Adam(G_net.parameters(),lr=0.001)
from torch.nn import BCELoss

# def print(x,y,c,figname):
#     plt.scatter(x,y,c=c)
#     plt.savefig(figname)
#     plt.close()

if __name__ == "__main__":
    from tqdm import tqdm
    normal = Normal(0,1)
    lossfunc = BCELoss()
    for epoch in tqdm(range(EPOCH)):
        for _ in range(UPDATEDTIME):
            realsample = normal.sample((BATCHSIZE,Distributionsize)).cuda()
            fakesample = torch.rand((BATCHSIZE,32)).cuda()
            fakeresult = G_net(fakesample).detach()
            D_optim.zero_grad()
            loss = 0
            realpredict = D_net(realsample)
            loss += lossfunc(realpredict[:,0],torch.ones_like(realpredict[:,0]))
            fakepredict = D_net(fakeresult)
            loss += lossfunc(fakepredict[:,0],torch.ones_like(fakepredict[:,1]))
            loss.backward()
        G_optimizer.zero_grad()
        noise = torch.rand((BATCHSIZE,32)).cuda()
        GeneratorDiscrim = D_net(G_net(noise))[:,0]
        loss = -lossfunc(GeneratorDiscrim,torch.ones_like(GeneratorDiscrim))
        # likelihood is p,BCEL return -log p for Generator we want p is samaller 
        loss.backward()
        G_optimizer.step()
        if epoch % 128 == 0:
            noise = torch.rand((1,32)).cuda()
            dis = G_net(noise).squeeze(0).detach().cpu()
            plt.hist(dis,bins=64)
            plt.savefig("./pic/figure"+str(int(epoch/128))+'.png')
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
