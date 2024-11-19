from numba import jit
from torch import nn
import numpy as np
import torch
import math
import csv
import sys

file = sys.argv[1]
nmues = int(sys.argv[2])

N = np.int64(500)
nstates = 500

x,y,a,b,ang=np.loadtxt(file,unpack=True,dtype="float64")

a = a[0]/2
b = b[0]/2
ang = np.radians(ang)

@jit(nopython=True,parallel=True)
def binary(x,y,a,b,ang,N):
    bin=np.zeros((1024,1024))
    npix = 1024
    for k in range(N):
        lx1 = math.floor((x[k]-a)*(npix-1))
        lx1 = 0 if lx1 < 0 else npix-1 if lx1 >= npix else lx1
        lx2 = math.ceil((x[k]+a)*(npix-1))
        lx2 = 0 if lx2 < 0 else npix-1 if lx2 >= npix else lx2
        ly1 = math.floor((y[k]-a)*(npix-1))
        ly1 = 0 if ly1 < 0 else npix-1 if ly1 >= npix else ly1
        ly2 = math.ceil((y[k]+a)*(npix-1))
        ly2 = 0 if ly2 < 0 else npix-1 if ly2 >= npix else ly2
        for i in range(lx1,lx2):
            px = i/(npix-1)
            for j in range(ly1,ly2):
                py = j/(npix-1)
                xr = (px-x[k])*np.cos(ang[k])+(py-y[k])*np.sin(ang[k])
                yr = -(px-x[k])*np.sin(ang[k])+(py-y[k])*np.cos(ang[k])
                if (((xr/a)**2 + (yr/b)**2) <= 1) :
                    bin[j,i]=1
    return(bin)

def frac(input):
    p = min(input.shape)
    n = 2**np.floor(np.log(p/2)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(1, n)
    counts = np.empty((len(sizes)))
    input = torch.Tensor(input).unsqueeze(0)
    i = 0
    for j in range(1,n): 
        m = nn.AvgPool2d(kernel_size=2, stride=2)
        output = m(input)
        output = torch.where((output != 0), torch.tensor([1.]), torch.tensor([0.]))
        count = torch.sum(output).item()
        counts[i] = count

        input = output
        i += 1
    x = np.log(sizes)
    y = np.log(counts)
    slope = np.dot(x-np.mean(x),y-np.mean(y))/np.sum((x-np.mean(x))**2)
    return(-slope)

f = open('datos/DFS.csv', 'a', newline='')
writer = csv.writer(f)
for i in range(nstates):
   xi,yi,angi=x[i*N:(i+1)*N],y[i*N:(i+1)*N],ang[i*N:(i+1)*N]
   writer.writerow([i+N*nmues,frac(binary(xi,yi,a,b,angi,N))])
f.close()