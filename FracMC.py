import csv
import sys
import math
import numpy as np
from numba import jit

file = sys.argv[1]
nmues = int(sys.argv[2])

N = np.int64(500)
nstates = 500

x,y,a,b,ang=np.loadtxt(file,unpack=True,dtype="float64")

#file = sys.argv[1]
#nmues = int(re.findall(r'\d+',file)[0])

@jit
def fracpygen(rows):
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

@jit(nopython=True,parallel=True)
def elipse(x,y,a,b,ang,N):
    bin=np.zeros((1024,1024))
    npix = 1024

    if a[0]>b[0]:
        c = a[0] 
    else :
        c = b[0]

    for k in range(N):
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
                if (((xr/a[k])**2 + (yr/b[k])**2) <= 1) :
                    bin[j,i]=1
    return(bin)

f = open('datos/DFS.csv', 'a', newline='')
writer = csv.writer(f)
for i in range(nstates):
   xi,yi,ai,bi,angi=x[i*N:(i+1)*N],y[i*N:(i+1)*N],a[i*N:(i+1)*N]/2,b[i*N:(i+1)*N]/2,ang[i*N:(i+1)*N]*np.pi/180
   writer.writerow([i+500*nmues,fracpygen(elipse(xi,yi,ai,bi,angi,N))])
f.close()