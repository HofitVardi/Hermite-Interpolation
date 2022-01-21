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

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0,0],data[:,0,1],data[:,0,2],marker="o",color="tab:blue")
ax.quiver(data[:,0,0],data[:,0,1],data[:,0,2],data[:,1,0],data[:,1,1],data[:,1,2],color="tab:blue",length=0.3,normalize=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()

mlr = data
for k in range(8):
    mlr = LR(1,mlr,1,periodic=True)
    xy = []                             # to determine transparency by distance from z axis.
    for i in range(len(mlr)):           # -"-
        xy.append(norm(mlr[i,0,0:2]))   # -"-
    d = np.max(np.array(xy))            # -"-
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(mlr)):
        ax.scatter(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],marker=".",color=str(0.5*((mlr[i,0,0]**2+mlr[i,0,1]**2)**0.5)/d),alpha=0.3)
        ax.quiver(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],mlr[i,1,0],mlr[i,1,1],mlr[i,1,2],color=str(0.5*((mlr[i,0,0]**2+mlr[i,0,1]**2)**0.5)/d),length=0.3,normalize=True,alpha=0.3)
    
    ax.scatter(data[:,0,0],data[:,0,1],data[:,0,2],marker="o",color="tab:blue")
    ax.quiver(data[:,0,0],data[:,0,1],data[:,0,2],data[:,1,0],data[:,1,1],data[:,1,2],color="tab:blue",length=0.3,normalize=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()