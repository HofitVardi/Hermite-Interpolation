import numpy as np
from numpy.linalg import norm

# method is the method of choosing the parameter of the vectors lenght.
# choose method=0 to use the sum of angles between vectors and the vector of differences.
# choose method=1 to use the angle between vectors.
# if <v_j,p1-p0>/||p1-p0||=-1 for j=0 and j=1 then alpha=0.

def BezierAv(t,pv0,pv1,method=0):
    p0,v0 = pv0[0],pv0[1]
    p1,v1 = pv1[0],pv1[1]

    if (method==0):
        cos_left = min(np.dot(v0,p1-p0)/(norm(v0)*norm(p1-p0)),1)   ## avoiding round-up errors
        cos_left = max(cos_left,-1)                                 ## -"""-
        cos_right = min(np.dot(v1,p1-p0)/(norm(v1)*norm(p1-p0)),1)  ## -"""-
        cos_right = max(cos_right,-1)                               ## -"""-
        theta = (np.arccos(cos_left)+np.arccos(cos_right))/4
    else:
        cos_theta = min(np.dot(v0,v1)/(norm(v0)*norm(v1)),1)        ## -"""-
        cos_theta = max(cos_theta,-1)                               ## -"""-
        theta = np.arccos(cos_theta)/4
    if (np.cos(theta)>10**(-12)):
        alpha = norm(p0-p1)/(3*np.cos(theta)**2)
    else:
        alpha = 0
    p = (1-t)**2*(1+2*t)*p0+3*alpha*t*(1-t)**2*v0+t**2*(3-2*t)*p1-3*alpha*t**2*(1-t)*v1
    v = 2*t*(t-1)*p0+alpha*(3*t**2-4*t+1)*v0+2*t*(1-t)*p1-alpha*t*(2-3*t)*v1
    if (norm(v)>10**(-12)):
        v = v/norm(v)
    return (np.array([p,v]))