import torchvision
import torch
from torchvision import datasets,transforms


data_train = datasets.MNIST(root = "./data/",
                            transform=transforms.ToTensor(),
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transforms.ToTensor(),
                           train = False)

from torch.utils.data import DataLoader
flatten = torch.nn.Flatten()
loader = DataLoader(data_train,batch_size=32)
for data,label in loader:
    print(data.shape)
    print(label.shape)
    print(flatten(data).shape)
    exit(    
    )
