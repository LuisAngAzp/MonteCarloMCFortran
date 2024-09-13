from numba.experimental import jitclass
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import random as ran
import numba as nb
import numpy as np
import contextlib
import math

@jitclass()
class MC:
    N: int
    rho: nb.float64
    T: nb.float64
    AR: nb.float64
    b_r: nb.float64

    nmc: int
    mcsi: int
    mcsc: int 
    nsub: int

    di: nb.float64
    dangc : nb.float64
    dc : nb.float64

    A1: nb.float64
    A2: nb.float64
    A3: nb.float64
    A4: nb.float64
    gamma: nb.float64
    krs : nb.float64

    rx: nb.float64[:]
    ry: nb.float64[:]
    ra: nb.float64[:]
    a: nb.float64
    b: nb.float64
    s: nb.float64

    nx: nb.float64[:]
    ny: nb.float64[:]
    na: nb.float64[:]

    tras: bool

    clu: nb.float64[:,:]

    DF: nb.float32[:]
    sizesample : int
    sampleratio : nb.float32

    def __init__(self):
        self.N = 500
        self.rho = 0.29
        self.T = 20.0+273.15
        self.AR = 3.0
        self.b_r = 1e-6

        self.nmc = 100
        self.mcsi = 4000
        self.mcsc = 1000
        self.nsub = 50

        self.di = 3.0
        self.dangc = 7.0
        self.dc = 8.0

        self.A1 = 20.01
        self.A2 = 0.394
        self.A3 = 4.086
        self.A4 = 3.851
        self.gamma = 72.75e-3 
        self.krs = 0

        self.rx = np.zeros(self.N)
        self.ry = np.zeros(self.N)
        self.ra = np.zeros(self.N)
        self.b = 0.0
        self.a = 0.0
        self.s = 0.0

        self.nx = np.zeros(self.N)
        self.ny = np.zeros(self.N)
        self.na = np.zeros(self.N)

        self.tras = False

        self.clu = np.zeros((self.N,self.N))

        self.DF = np.zeros(self.nmc,dtype=np.float32)
        self.sizesample = 10
        self.sampleratio = 1/self.sizesample
    
    def start(self):
        self.s = 2.0*math.sqrt(self.rho/(self.AR*np.pi*self.N))
        self.b = self.s/2
        self.a = self.AR*(self.b)
        self.di = self.di*self.s
        self.krs = self.b_r/self.b

        a = self.a
        b = self.b
        AR = self.AR
        pi = np.pi

        i = 0
        while (i<=self.N-1):
            self.tras = True
            count = 0
            while (self.tras == True):
                count += 1
                self.rx[i] = ran.uniform(0,1)
                self.ry[i] = ran.uniform(0,1)
                self.ra[i] = 180*ran.uniform(0,1)

                if (i == 0): 
                    i += 1
                    break
                
                j = 0
                while (j<=i-1):
                    a1 = self.ra[i]*pi/180
                    a2 = self.ra[j]*pi/180

                    X = self.rx[j] - self.rx[i]
                    Y = self.ry[j] - self.ry[i]

                    if (X > 0.5):
                        X += -1
                    elif (X < -0.5):
                        X += 1
                    
                    if (Y > 0.5):
                        Y += -1
                    elif (Y < -0.5):
                        Y += 1

                    GGG = 2.00 +(AR-(1/AR))**2*(np.sin((a2-a1)))**2

                    F1=1+GGG-(X*np.cos(a1)+Y*np.sin(a1))**2/a**2-(X*np.sin(a1)-Y*np.cos(a1))**2/b**2
                    F2=1+GGG-(X*np.cos(a2)+Y*np.sin(a2))**2/a**2-(X*np.sin(a2)-Y*np.cos(a2))**2/b**2

                    FI=4*(F1**2-3*F2)*(F2**2-3*F1)-(9-F1*F2)**2

                    self.tras = not ((FI>0) and ((F1<0) or (F2<0)))
                    
                    if (self.tras == True):
                        break
                    j += 1
            if (j == i):
                i += 1

    def ellipses(self,ang1,ang2,ang3):
        au = complex(1,0)

        cs1=np.cos(ang1);sn1=np.sin(ang1)
        cs2=np.cos(ang2);sn2=np.sin(ang2)
        cs3=np.cos(ang3);sn3=np.sin(ang3)
        k1d=np.cos(ang3-ang1)
        k2d=np.cos(ang3-ang2)
        k1k2=np.cos(ang2-ang1)

        a = self.a
        b = self.b
        AR = self.AR
        e=1-(1/AR)**2 

        eta=AR-1
        a11=1+0.5*(1+k1k2)*(eta*(2+eta)-e*(1+eta*k1k2)**2)
        a12=0.5*np.sqrt(1-k1k2**2)*(eta*(2+eta)+e*(1-eta**2*k1k2**2))
        a22=1+0.5*(1-k1k2)*(eta*(2+eta)-e*(1-eta*k1k2)**2)

        lam1=0.5*(a11+a22)+0.5*np.sqrt((a11-a22)**2+4*a12**2)
        lam2=0.5*(a11+a22)-0.5*np.sqrt((a11-a22)**2+4*a12**2)

        b2p=1/np.sqrt(lam1)
        a2p=1/np.sqrt(lam2)

        deltap=(a2p/b2p)**2-1

        if (abs(k1k2) == 1):
            if (a11>a22):
                kpmp=1/np.sqrt(1-e*k1d**2)*k1d*(1/AR)
            else:
                kpmp=(sn3*cs1-cs3*sn1)/np.sqrt(1-e*k1d**2)
        else:
            kpmp=(a12/np.sqrt(1+k1k2)*(k1d/AR+k2d+(1/AR-1)*k1d*k1k2)+(lam1-a11)
                    /np.sqrt(1-k1k2)*(k1d/AR-k2d-(1/AR-1)*k1d*k1k2))/np.sqrt(2*(a12**2+(lam1-a11)**2)*(1-e*k1d**2))
        if ((kpmp == 0) or deltap == 0):
            Rc=a2p+1
        else:
            t=1/kpmp**2-1
            A=-1/b2p**2*(1+t)
            B=-2/b2p*(1+t+deltap)
            C=-t-(1+deltap)**2+1/b2p**2*(1+t+deltap*t)
            D=2/b2p*(1+t)*(1+deltap)
            E=(1+t+deltap)*(1+deltap)

            alpha=-3/8*(B/A)**2+C/A
            beta=(B/A)**3/8-(B/A)*(C/A)/2+D/A
            gamma=-3/256*(B/A)**4+C/A*(B/A)**2/16-(B/A)*(D/A)/4.+E/A

            if (beta == 0):
                qq = -B/4/A+(-alpha+(alpha**2-4*gamma*au)**(0.5)/2)**(0.5)
            else:
                P=-alpha**2/12-gamma
                Q=-alpha**3/108+gamma*alpha/3-beta**2/8
                U=(-0.5*Q+(Q**2/4+P**3/27*au)**(0.5))**(1/3.0)

                if (abs(U) != 0):
                    y=-5/6*alpha+U-P/3/U
                else:
                    y=-5/6*alpha-Q**(1/3)
                
                qq=-B/4/A+0.5*((alpha+2*y)**(0.5)+(-(3*alpha+2*y+2*beta/(alpha+2*y)**(0.5)))**(0.5))

            Rc=complex(((qq**2-1)/deltap*(1+b2p*(1+deltap)/qq)**2+(1-(qq**2-1)/deltap)*(1+b2p/qq)**2)**0.5)
        
        dist=Rc.real*self.b_r/(1-e*k1d**2)**0.5
        return(dist)
    
    def ET(self):
        U = 0
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                X = self.rx[j] - self.rx[i]
                Y = self.ry[j] - self.ry[i]

                if (X > 0.5):
                        X += -1
                elif (X < 0.5):
                    X += 1
                
                if (Y > 0.5):
                    Y += -1
                elif (Y < 0.5):
                    Y += 1
                
                RR = X*X + Y*Y

                if (RR <= 25*self.a**2):
                    ang1 = self.ra[i]*np.pi/180
                    ang2 = self.ra[j]*np.pi/180

                    ang3 = np.arctan(Y/X)
                    dist = self.ellipses(ang1,ang2,ang3)

                    U = U - self.gamma*self.b_r**2*self.A1*np.cos(2*ang1+2*ang2)*(self.b_r/(np.sqrt(RR)-self.A2*dist+self.A3))**self.A4
    
    def EP(self,i,x,y,a):
        U = 0.0
        for j in range(self.N):
            if (j != i):
                X = (self.rx[j] - x)
                Y = (self.ry[j] - y)
                print("Caja",X,Y,5*self.a)

                if (X > 0.5):
                    X += -1
                elif (X < 0.5):
                    X += 1
                if (Y > 0.5):
                    Y += -1
                elif (Y < 0.5):
                    Y += 1
                
                X = X*self.krs
                Y = Y*self.krs
                RR = X*X + Y*Y
                
                print(X,Y,5*(self.a*self.krs))
                if (RR < 25*(self.a*self.krs)**2):
                    ang1 = np.arctan((X*np.cos(a*np.pi/180)+Y*np.sin(a*np.pi/180))/RR)
                    ang2 = np.arctan((X*np.cos(self.ra[j]*np.pi/180)+Y*np.sin(self.ra[j]*np.pi/180))/RR)

                    ang3 = np.arctan(Y/X)
                    dist = self.ellipses(ang1,ang2,ang3)

                    #phi = np.cos(2*ang1+2*ang2)
                    U = U - np.cos(2*ang1+2*ang2)*self.A1*(self.b_r/(RR**0.5-self.A2*dist+self.A3))**self.A4
                    #print(U)
        return(U)
    
    def over(self,i,x,y,ang):
        for j in range(0,self.N):
            if (j != i):
                a1 = ang*np.pi/180
                a2 = self.ra[j]*np.pi/180
                AR = self.AR
                a = self.a 
                b = self.b

                X = self.rx[j] - x
                Y = self.ry[j] - y

                if (X > 0.5):
                        X += -1
                elif (X < -0.5):
                    X += 1
                
                if (Y > 0.5):
                    Y += -1
                elif (Y < -0.5):
                    Y += 1

                GGG = 2.00 +(AR-(1/AR))**2*(np.sin((a2-a1)))**2

                F1=1+GGG-(X*np.cos(a1)+Y*np.sin(a1))**2/a**2-(X*np.sin(a1)-Y*np.cos(a1))**2/b**2
                F2=1+GGG-(X*np.cos(a2)+Y*np.sin(a2))**2/a**2-(X*np.sin(a2)-Y*np.cos(a2))**2/b**2

                FI=4*(F1**2-3*F2)*(F2**2-3*F1)-(9-F1*F2)**2

                log = not ((FI>0) and ((F1<0) or (F2<0)))

                #print(i,j,log,FI,F1,F2)
                if (log == True):
                    break
        return(log)

    def fracpygen(self,rows):
        h, w = rows.shape
        
        m = np.int64(np.floor(np.log10(w)/np.log10(2))-1)

        cnts = np.zeros((m), dtype=np.float64)
        for lev in np.arange(m):
            cnt = 0
            block_size = 2**lev
            for j in np.arange(int(w/(block_size))):
                for i in np.arange(int(h/block_size)):
                    posj = j*block_size
                    posi = i*block_size
                    posjf = posj + block_size
                    posif = posi + block_size
                    if posjf > w: posjf = w
                    if posif > h: posif = h
                    cnt = cnt + rows[posj:posjf, posi:posif].any()
            cnts[lev]=cnt
        data = np.array([(1/2**(k),cnts[k]) for k in range(m)],dtype="float64")
        xs = np.log10(data[:,0])
        ys = np.log10(data[:,1])
        
        X = np.vstack((xs,np.ones(len(xs)))).T

        b1 = np.linalg.lstsq(X,ys)[0]
        return(b1[0])
    
    def elipse(self):
        x = self.rx
        y = self.ry
        ang = self.ra
        a = self.a
        b = self.b
        
        bin = np.zeros((1024,1024))
        npix = 1024

        c = a 

        for k in range(self.N):
            lx1 = math.floor((x[k]-c)*(npix-1))
            lx1 = 0 if lx1 <= 0 else npix if lx1 >= npix else lx1
            lx2 = math.ceil((x[k]+c)*(npix-1))
            lx2 = 0 if lx2 <= 0 else npix if lx2 >= npix else lx2
            ly1 = math.floor((y[k]-c)*(npix-1))
            ly1 = 0 if ly1 <= 0 else npix if ly1 >= npix else ly1
            ly2 = math.ceil((y[k]+c)*(npix-1))
            ly2 = 0 if ly2 <= 0 else npix if ly2 >= npix else ly2

            print
            for i in range(lx1,lx2):
                px = i/(npix-1)
                for j in range(ly1,ly2):
                    py = j/(npix-1)
                    xr = (px-x[k])*np.cos(ang[k])+(py-y[k])*np.sin(ang[k])
                    yr = -(px-x[k])*np.sin(ang[k])+(py-y[k])*np.cos(ang[k])
                    if (((xr/a)**2 + (yr/b)**2) <= 1) :
                        bin[j,i]=1
        return(bin)
        
    def moves(self):
        #f = open('datos/DFSimulacion.csv', 'a', newline='')
        #writer = csv.writer(f)
        
        kb = 1.380649e-23
        ncount = 0
        while (ncount < self.nmc):
            ei = ran.randint(0,self.N-1)

            ncount += 1

            print(ncount)

            nx = self.rx[ei] + self.di*(ran.uniform(0,1)-0.5)
            ny = self.ry[ei] + self.di*(ran.uniform(0,1)-0.5)
            na = self.ra[ei] + 180*(ran.uniform(0,1)-0.5)

            if (nx < 0):
                nx = nx + 1
            elif (nx > 1):
                nx = nx - 1
            if (ny < 0):
                ny = ny + 1
            elif (ny > 1):
                ny = ny - 1
            if (na < 0):
                na = na + 180
            elif (na > 180):
                na = na - 180

            log = self.over(ei,nx,ny,na)

            if (log == False):
                UOLD = self.EP(ei,self.rx[ei],self.ry[ei],self.ra[ei])
                UNEW = self.EP(ei,nx,ny,na)
                DU = UNEW - UOLD 
                if (DU < 0):
                    self.rx[ei] = nx
                    self.ry[ei] = ny
                    self.ra[ei] = na
                else:
                    rnd = ran.random()
                    DU = -DU*self.gamma*self.b_r**2/(self.T*kb)
                    exp = np.exp(DU)
                    print(DU,exp,UOLD,UNEW)
                    if (exp > rnd):
                        self.rx[ei] = nx
                        self.ry[ei] = ny
                        self.ra[ei] = na
            if ((ncount % self.sizesample) == 0):
                bin = self.elipse()
                self.DF[int(ncount/self.sizesample)-1] = self.fracpygen(bin)
            
Monte = MC()
Monte.start()
print("hOLA")

def elipse(x,y,a,b,ang,N):
   fig = plt.figure(figsize=(5,5))
   ax = fig.add_subplot(111, aspect='equal')
   ax.set_xlim(0,1)
   ax.set_ylim(0,1)
   for j in range(N):
      ax.add_artist(Ellipse(xy=(x[j],y[j]), width=2*a, height=2*b, angle= ang[j],facecolor="black"))
   plt.axis('off')
   plt.show()

elipse(Monte.rx,Monte.ry,Monte.a,Monte.b,Monte.ra,Monte.N)
#with open('myfile.txt','w') as myfile:
#    with contextlib.redirect_stdout(myfile):
#        Monte.moves()
Monte.moves()

elipse(Monte.rx,Monte.ry,Monte.a,Monte.b,Monte.ra,Monte.N)
