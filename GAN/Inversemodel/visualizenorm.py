from matplotlib import pyplot as plt
import numpy as np

mu = 0
sigma = 1
N = 1024 * 1024
data = np.random.normal(mu,sigma,N)
plt.hist(data,bins=1000)
plt.savefig("normal.png")

