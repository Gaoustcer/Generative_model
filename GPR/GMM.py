import numpy as np

from random import random

import matplotlib.pyplot as plt
N = 1024
K = 8
import numpy as np

def Gaussiandistance(x:np.ndarray,y:np.ndarray):
    distance = np.sum((x - y)**2)
    return np.exp(-distance/2)

def gaussiankernel(x:np.ndarray,y:np.ndarray):
    '''
    X:n*k matrix y m*k matrix you need to calculate its gaussian kernel
    '''
    kernel = np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            kernel[i][j] = Gaussiandistance(x[i],y[j])
    # pass
    return kernel


Data = np.random.random((N,K)) - 0.5
label = np.sin(np.sum(Data,-1))
plt.scatter(x = np.sum(Data,-1),y = label,s=0.1,c='r')
Dataaverage = np.average(Data,0)
# Data = Data - Dataaverage
# plt.savefig("GMM.png")
N_test = 1024
XData = np.random.random((N_test,K)) - 0.5
Xlabel = np.sin(np.sum(XData,-1))
plt.scatter(x=np.sum(XData,-1),y=Xlabel,c='g',s=0.5)
# XData = XData - np.average(XData,0)
Cov = gaussiankernel(XData,Data)
# Cov = XData.dot(Data.T)
DataCov = gaussiankernel(Data,Data)
# DataCov = Data.dot(Data.T)
print("Cov shape is",Cov.shape,"DataCov shape is",DataCov.shape)
# print("Cov shape",Cov.shape,"DataCov shape",DataCov.shape)
DataCov = np.linalg.inv(DataCov)
plt.savefig("Baseline.png")
pred = np.matmul(Cov,DataCov).dot(label)
plt.scatter(x=np.sum(XData,-1),y=pred,c='b',s=0.5)
plt.savefig("GMM.png")
