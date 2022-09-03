# from  .Discriminator import Discriminator
# from .Generator import Generator
from Generate_model.Discriminator import Discriminator
from Generate_model.Generator import Generator
# from Generate_model.Generator import Generator
# from .Generator import Generator
from torch.distributions import Normal
BATCHSIZE = 1024
import torch
import matplotlib.pyplot as plt
def print(x,y,c,figname):
    plt.scatter(x,y,c=c)
    plt.savefig(figname)

if __name__ == "__main__":
    import numpy as np
    c = ['r','g']
    figname = ['fig1.png','fig2.png']
    for i in range(2):
        x = np.random.random(32)
        y = np.random.random(32)
        print(x,y,c[i],figname[i])
