from deduction.VAE import VAE
from model.model import Encoder,Decoder
import sys
import os
sys.path.append("..")
from dataset import data_train,data_test


EPOCH = 32
Variational_deduction = VAE(Encoder().cuda(),Decoder().cuda(),data_train,data_test)
if __name__ == "__main__":
    path = "pictureonlyBCE"
    for epoch in range(EPOCH):
        Variational_deduction.train()
        Variational_deduction.valid(path+"/Validation{}".format(epoch))
        Variational_deduction.deduction(path+"/Generate{}".format(epoch))