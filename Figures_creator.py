import numpy as np
from BezierAv import BezierAv
from modified_schemes import LR
from LinearLR import LR as LLR
from hermite_generator import hermite_generator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm


def create_fig_(k):
    if (k==1): ###  Figure 1  ###
        angles = np.array([[0.6*np.pi,-0.4*np.pi],[0.4*np.pi,0.4*np.pi]]) # different pairs of angles for different examples.
        for a in angles:
            theta = a[0]
            phi = a[1]
            pv0 = np.array([[0,0],[np.cos(theta),np.sin(theta)]]) 
            pv1 = np.array([[4,0],[np.cos(phi),np.sin(phi)]])

            curve = []
            for t in np.linspace(0,1,100):
                curve.append(BezierAv(t,pv0,pv1,method=0))
            curve = np.array(curve)

            av = BezierAv(0.5,pv0,pv1,method=0)

            data = np.array([pv0,pv1])
            p0,v0 = pv0[0],pv0[1]
            p1,v1 = pv1[0],pv1[1]

            cos_left = min(np.dot(v0,p1-p0)/(norm(v0)*norm(p1-p0)),1)   ## avoiding round-up errors
            cos_left = max(cos_left,-1)                                 ## -""-
            cos_right = min(np.dot(v1,p1-p0)/(norm(v1)*norm(p1-p0)),1)  ## -""-
            cos_right = max(cos_right,-1)                               ## -""-
            theta = (np.arccos(cos_left)+np.arccos(cos_right))/4
            if (np.cos(theta)>10**(-12)):
                alpha = norm(p0-p1)/(3*np.cos(theta)**2)
            else:
                alpha = 0
            control_polygon = np.array([p0,p0+alpha*v0,p1-alpha*v1,p1])

            plt.plot(control_polygon[:,0],control_polygon[:,1],color="gray",linestyle="dashed",alpha=0.5)
            plt.plot(curve[:,0,0],curve[:,0,1],color="gray",alpha=0.5)
            plt.scatter(data[:,0,0],data[:,0,1],marker='o',color="black")
            plt.quiver(data[:,0,0],data[:,0,1],data[:,1,0],data[:,1,1],angles='xy', scale_units='xy', scale=1,color="black")
            plt.scatter(av[0,0],av[0,1],marker='o',color="tab:blue")
            plt.quiver(av[0,0],av[0,1],av[1,0],av[1,1],angles='xy', scale_units='xy', scale=1,color="tab:blue")
            
            data = np.array([pv0,av,pv1])
            plt.xlim(-1,5)
            plt.ylim(-2,3)
            plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False) 
            plt.title("Bezier Average")
            plt.show()


    if (k==2):  ###  Figure 2  ###
        angles = np.array([[0.6*np.pi,-0.4*np.pi],[0.4*np.pi,0.4*np.pi]]) # different pairs of angles for different examples.
        for a in angles:
            theta = a[0]
            phi = a[1]
            pv0 = np.array([[0,0],[np.cos(theta),np.sin(theta)]]) 
            pv1 = np.array([[4,0],[np.cos(phi),np.sin(phi)]])

            curve = []
            for t in np.linspace(0,1,100):
                curve.append(BezierAv(t,pv0,pv1,method=0))
            curve = np.array(curve)

            av = BezierAv(0.5,pv0,pv1,method=0)

            data = np.array([pv0,pv1])
            for n in np.array([0,1,2,3,6]):
                lr = LR(1,data,n,periodic=False)

                plt.plot(lr[:,0,0],lr[:,0,1],marker="o",markersize="4",color="black")
                plt.quiver(lr[:,0,0],lr[:,0,1],lr[:,1,0],lr[:,1,1],angles='xy', scale_units='xy', scale=1,color="black",width=0.005)

                plt.xlim(-0.4,4.7)
                plt.ylim(-1.5,2.5)
                plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                for pos in ['right', 'top', 'bottom', 'left']:
                    plt.gca().spines[pos].set_visible(False)
                plt.title(str(n)+" iterations")
                plt.show()


    if (k==3):  ###  Figure 3  ###
        data = np.array([[[0,0,0],[-1/2**0.5,1/2**0.5,0]],[[2,0,8],[1/2**0.5,1/2**0.5,0]]])
        iterations = []
        d = 0
        for n in np.array([0,1,2,3,6]):
            mlr = LR(1,data,n,periodic=False)
            xy = []                             # to determine transparency by distance from z axis.
            for i in range(len(mlr)):           # -"-
                xy.append(norm(mlr[i,0,0:2]))   # -"-
            s = np.max(np.array(xy))            # -"-
            if(s>d):                            # -"-
                d = s                           # -"-
            iterations.append([mlr,n])
        iterations = np.array(iterations)

        for iter in iterations:
            mlr = iter[0]
            n = iter[1]
            fig = plt.figure()
            ax = Axes3D(fig)
            for i in range(len(mlr)):
                ax.scatter(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],marker="o",color=str(0.5*((mlr[i,0,0]**2+mlr[i,0,1]**2)**0.5)/d))
                ax.quiver(mlr[i,0,0],mlr[i,0,1],mlr[i,0,2],mlr[i,1,0],mlr[i,1,1],mlr[i,1,2],color=str(0.5*((mlr[i,0,0]**2+mlr[i,0,1]**2)**0.5)/d))
            ax.set_xlim(-1,3)
            ax.set_ylim(-1,1)
            ax.set_zlim(0,9)
            ax.view_init(10,-45)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            save = True
            if (save):
                plt.savefig("3D"+str(n)+".pdf")
            ax.set_title("n="+str(n))
            plt.show()


    if (k==4):  ###  Figure 4  ###
        # 2D spiral.
        m = 1
        spiral = []
        for t in np.linspace(0,10*np.pi,1000):
            v = np.array([m*np.cos(t)-m*t*np.sin(t),m*np.sin(t)+m*t*np.cos(t)])
            v = v/norm(v)
            spiral.append([[m*t*np.cos(t),m*t*np.sin(t)],v])
        spiral = np.array(spiral)

        data_length = np.array([4,8,16])
        for n in data_length:
            data = []
            for t in np.linspace(0,10*np.pi,n):
                v = np.array([m*np.cos(t)-m*t*np.sin(t),m*np.sin(t)+m*t*np.cos(t)])
                v = v/norm(v)
                data.append([[m*t*np.cos(t),m*t*np.sin(t)],v])
            data = np.array(data)

            A = LR(1,data,8,average=1,periodic=False)   # Approximation by alternative average
            N = LR(1,data,8,periodic=False)             # Approximation by IHB

            plt.plot(N[:,0,0],N[:,0,1],color="black")
            plt.plot(A[:,0,0],A[:,0,1],color="tab:blue")
            plt.plot(spiral[:,0,0],spiral[:,0,1],color="gray",alpha=0.5)
            plt.scatter(data[:,0,0],data[:,0,1],marker="o",color="gray",alpha=0.5)
            plt.quiver(data[:,0,0],data[:,0,1],data[:,1,0],data[:,1,1],angles='xy', scale_units='xy',color="gray",width=0.005)
            plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)
            plt.legend(["IHB","Alternative average","Archimedean Spiral"])
            plt.title("n="+str(n))
            plt.show()

        # 3D spiral.
        a = 0
        b = 0
        m = 5
        k = 1
        for n in np.array([4,8,16]):
            curve = []
            for t in np.linspace(0,10*np.pi,1000):
                v = np.array([m*np.cos(t)-m*t*np.sin(t),m*np.sin(t)+m*t*np.cos(t),k*m])
                v = v/norm(v)
                curve.append([[m*t*np.cos(t),m*t*np.sin(t),k*m*t],v])
            curve = np.array(curve)

            data = []
            for t in np.linspace(0+a,10*np.pi+b,n):
                v = np.array([m*np.cos(t)-m*t*np.sin(t),m*np.sin(t)+m*t*np.cos(t),k*m])
                v = v/norm(v)
                data.append([[m*t*np.cos(t),m*t*np.sin(t),k*m*t],v])
            data = np.array(data)
            
            N = LR(1,data,8,periodic=False) # Approximation by IHB
            A = LR(1,data,8,1,False)        # Approximation by alternative average

            ax = plt.axes(projection='3d')
            ax.plot(N[:,0,0],N[:,0,1],N[:,0,2],color="black")
            ax.plot(A[:,0,0],A[:,0,1],A[:,0,2],color="tab:blue")
            ax.plot(curve[:,0,0],curve[:,0,1],curve[:,0,2],color="gray",alpha=0.5)
            ax.scatter(data[:,0,0],data[:,0,1],data[:,0,2],marker="o",color="gray",alpha=0.5)
            ax.quiver(data[:,0,0],data[:,0,1],data[:,0,2],50*data[:,1,0],50*data[:,1,1],50*data[:,1,2],color="gray",alpha=0.8)
            ax.view_init(30,-60)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.legend(["IHB","Alternative average","true curve"])
            plt.title("n="+str(n))
            plt.show()


    if (k==5):  ###  Figure 5  ###
        m = 12
        a = 0
        for step in np.array([2,10/9,1,2/3,0.05]):
            data = []
            for t in np.arange(a,m*np.pi,step*np.pi):
                v = np.array([1,np.cos(t)])
                v = v/norm(v)
                data.append([[t,np.sin(t)],v])
            data = np.array(data)

            mlr = LR(1,data,6,average=0,periodic=False) # Approximation by IHB

            bez = []  ## p.w cubic interpolation
            for i in range(len(data)-1):
                b0 = data[i,0]
                b1 = b0 + data[i,1]/3
                b3 = data[i+1,0]
                b2 = b3 - data[i+1,1]/3
                for t in np.linspace(0,1,10**3):
                    bez.append((1-t)**3*b0+3*t*(1-t)**2*b1+3*t**2*(1-t)*b2+t**3*b3)
            bez = np.array(bez)

            while (mlr[0,0,0]<2*np.pi):
                mlr = mlr[1:]
            while (bez[0,0]<2*np.pi):
                bez = bez[1:]
            while (data[0,0,0]<2*np.pi):
                data = data[1:]
                
            while (mlr[len(mlr)-1,0,0]>10*np.pi):
                mlr = mlr[:len(mlr)-1]
            while (bez[len(bez)-1,0]>10*np.pi):
                bez = bez[:len(bez)-1]
            while (data[len(data)-1,0,0]>10*np.pi):
                data = data[:len(data)-1]

            plt.plot(mlr[:,0,0],mlr[:,0,1],color="black")
            plt.plot(bez[:,0],bez[:,1],color="tab:red")
            plt.plot(np.linspace(2*np.pi,10*np.pi,1000),np.sin(np.linspace(2*np.pi,10*np.pi,1000)),color="gray",alpha=0.5)
            plt.scatter(data[:,0,0],data[:,0,1],color="gray",alpha=0.5)
            plt.quiver(data[:,0,0],data[:,0,1],data[:,1,0],data[:,1,1],angles='xy', scale_units='xy',color="gray",width=0.005)
            plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)      
            plt.legend(["IHB","P.W. Bezier","True curve"])
            plt.title("h="+str(step)+"*Pi")
            plt.show()


    if (k==6):  ###  Figure 6  ###
        m = 20
        for step in np.array([[2,np.pi/2],[10/9,0.3],[1,1],[2/3,0]]):
            a = step[1]-4*np.pi
            step = step[0]
            data = []
            for t in np.arange(a,m*np.pi,step*np.pi):
                v = np.array([1,-np.sin(t)])
                v = v/norm(v)
                data.append([[t,np.cos(t)],v])
            data = np.array(data)
        
            lr3_real = LR(3,data[1:len(data)-1],periodic=False)
            data = hermite_generator(data[:,0],False,False)
            llr3 = LLR(3,data,periodic=False)
            lr3 = LR(3,data,periodic=False)
            
            while (lr3[0,0,0]<5.5*np.pi):
                lr3 = lr3[1:]
            while (llr3[0,0,0]<5.5*np.pi):
                llr3 = llr3[1:]
            while (lr3_real[0,0,0]<5.5*np.pi):
                lr3_real = lr3_real[1:]
            while (data[0,0,0]<5.5*np.pi):
                data = data[1:]
                
            while (lr3[len(lr3)-1,0,0]>13.5*np.pi):
                lr3 = lr3[:len(lr3)-1]
            while (llr3[len(llr3)-1,0,0]>13.5*np.pi):
                llr3 = llr3[:len(llr3)-1]
            while (lr3_real[len(lr3_real)-1,0,0]>13.5*np.pi):
                lr3_real = lr3_real[:len(lr3_real)-1]
            while (data[len(data)-1,0,0]>13.5*np.pi):
                data = data[:len(data)-1]
            

            plt.plot(lr3_real[:,0,0],lr3_real[:,0,1],color="black", linewidth=1)
            plt.plot(lr3[:,0,0],lr3[:,0,1],color="black",linestyle="dashed", linewidth=1)
            plt.plot(llr3[:,0,0],llr3[:,0,1],color="orange")
            plt.plot(np.linspace(5.5*np.pi,13.5*np.pi,1000),np.cos(np.linspace(5.5*np.pi,13.5*np.pi,1000)),color="gray",alpha=0.5)
            plt.scatter(data[:,0,0],data[:,0,1],color="gray",alpha=0.5)
            plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)
            save = False
            if (save):
                plt.savefig("naive_vectors"+str(step)+"pi.eps")
            plt.legend(["MLR3","MLR3 over artificially generated Hermite data","Linear LR3","True curve"])
            plt.title("h="+str(step)+"pi")
            plt.show()


    if (k==7):  ###  Figure 7  ###
        def sample(t):  # Sample a curve's value and tangent direction at the parmeter t.
            p = np.array([t,np.sin(t)])     # Change p and v with accordance to your chosen curve.
            v = np.array([1,np.cos(t)])
            return np.array([p,v/norm(v)])

        ERR = []    # list of maximal errors in the approximation by IHB.
        ERR_L = []  # list of maximal errors in the approximation by Linear LR1.
        low = 10    # The largest distance between samples paramter-wise is (b-a)/low.
        high = 100  # The smallest distance between samples paramter-wise is (b-a)/high.
        a = 0
        b = 1
        for h in np.linspace((b-a)/high,(b-a)/low,100):
            data = []           # Initialize data with Hermite samples of a curve to your choice (defined in "sample")
                                # at equidistant parmeters with distance h over [a,b].
            for t in np.arange(a,b,h):
                data.append(sample(t))
            data = np.array(data)
            mlr = LR(1,data,6,periodic=False)   # IHB approximation
            llr = LLR(1,data,6,periodic=False)  # Linear LR1 approximation

            err = []    # list of errors in the approximation by IHB
            err_l = []  # list of errors in the approximation by Linear LR1
            for i in range(len(mlr)):
                err.append(np.abs(mlr[i,0,1]-sample(mlr[i,0,0])[0,1]))
                err_l.append(np.abs(llr[i,0,1]-sample(llr[i,0,0])[0,1]))
            err = np.array(err)
            err_l = np.array(err_l)
            
            ERR.append(np.max(err))
            ERR_L.append(np.max(err_l))

        ERR = np.array(ERR)
        ERR_L = np.array(ERR_L)
        plt.loglog(np.linspace((b-a)/high,(b-a)/low,len(ERR)),ERR,color="black")
        plt.loglog(np.linspace((b-a)/high,(b-a)/low,len(ERR_L)),ERR_L,color="black",linestyle="dashed")
        plt.grid(True,which="both",alpha=0.2)
        plt.legend(["IHB","Linear LR1"])
        plt.show()