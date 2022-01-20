from os import error
import numpy as np
from numpy.linalg.linalg import norm
from BezierAv import BezierAv
import matplotlib.pyplot as plt


# choose average=j to use Bezier average with alpha=j
def AvScheme(f,num_iterations=1,average=0,periodic=True):
    if (num_iterations==0):
        return f
    else:
        new = []
        for i in range(len(f)-1):
            new.append(f[i])
            new.append(BezierAv(0.5,f[i],f[i+1],average))
        new.append(f[len(f)-1])
        if (periodic):
            new.append(BezierAv(0.5,f[len(f)-1],f[0],average))
        return AvScheme(np.array(new),num_iterations-1,average,periodic)


def smoothing(f,num_iterations,average=0,periodic=True):
    if (num_iterations<1):
        return f
    else:
        if (len(f)<2):
            error("not enough data")
        new = []
        for i in range(len(f)-1):
            new.append(BezierAv(0.5,f[i],f[i+1],average))
        if (periodic):
            new.append(BezierAv(0.5,f[len(f)-1],f[0],average))
        return smoothing(np.array(new),num_iterations-1,average,periodic)



# Modified Lane-Risenfeld with m-1 smoothing steps
# choose average=j to use Bezier average with alpha=j
def LR(m,f,iterations=8,average=0,periodic=True):
    for i in range(len(f)):
        f[i,1] = f[i,1]/norm(f[i,1])
    if (iterations==0):
        return f
    else:
        f = AvScheme(f,1,average,periodic) #refinement step
        f = smoothing(f,m-1,average,periodic)
        return LR(m,f,iterations-1,average,periodic)