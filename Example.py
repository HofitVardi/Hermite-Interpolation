from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from numpy import random
from modified_schemes import LR
import matplotlib.pyplot as plt
from numpy.linalg import norm


# 3D interpolation from random samples
n = 10
data = []
for i in range(n):
    p = random.uniform(-1,1,3)
    v = random.uniform(-1,1,3)
    v = v/norm(v)
    data.append([p,v])
data = np.array(data)

for iter in range(8):
    mlr = LR(1,data,iter,periodic=False)
    xy = []                             # to determine transparency by distance from z axis.
    for i in range(len(mlr)):           # -"-
        xy.append(norm(mlr[i,0,0:2]))   # -"-
    s = np.max(np.array(xy)) 
            # -"-
    d=s
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(mlr)):
        ax.scatter(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],marker=".",color=str(0.5*((mlr[i,0,0]**2+mlr[i,0,1]**2)**0.5)/d),alpha=0.3)
        ax.quiver(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],mlr[i,1,0],mlr[i,1,1],mlr[i,1,2],color=str(0.5*((mlr[i,0,0]**2+mlr[i,0,1]**2)**0.5)/d),length=0.3,normalize=True,alpha=0.3)
    mlr = data
    for i in range(len(mlr)):
        ax.scatter(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],marker="o",color="tab:blue")
        ax.quiver(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],mlr[i,1,0],mlr[i,1,1],mlr[i,1,2],color="tab:blue",length=0.3,normalize=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()