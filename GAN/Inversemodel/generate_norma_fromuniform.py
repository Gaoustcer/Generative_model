from random import random
from math import log
from math import pi
from math import sqrt
mu = 0
sigma = 1
N = 1024 ** 2
data = []
for _ in range(N):
    P = random()
    possibility = 1-(2*P - 1)**2
    u2 = -log(possibility)
    if P > 0.5:
        u = sqrt(u2)
    else:
        u = -sqrt(u2)
    data.append(u)
import matplotlib.pyplot as plt
plt.hist(data,bins=1000)
plt.savefig("Generate_from_uniform.png")