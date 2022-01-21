from os import error
import numpy as np


def AvScheme(f,num_iterations=1,periodic=True):
    if (num_iterations==0):
        return f
    else:
        new = []
        for i in range(len(f)-1):
            new.append(f[i])
            new.append((f[i]+f[i+1])/2)
        new.append(f[len(f)-1])
        if (periodic):
            new.append((f[len(f)-1]+f[0])/2)
        return AvScheme(np.array(new),num_iterations-1,periodic)


def smoothing(f,num_iterations,periodic=True):
    if (num_iterations<1):
        return f
    else:
        if (len(f)<2):
            error("not enough data")
        new = []
        for i in range(len(f)-1):
            new.append((f[i]+f[i+1])/2)
        if (periodic):
            new.append((f[len(f)-1]+f[0])/2)
        return smoothing(np.array(new),num_iterations-1,periodic)



# Lane-Risenfeld with m-1 smoothing steps.
# f is the data to refine.
def LR(m,f,iterations=8,periodic=True):
    if (iterations==0):
        return f
    else:
        f = AvScheme(f,1,periodic) #refinement step
        f = smoothing(f,m-1,periodic)
        return LR(m,f,iterations-1,periodic)