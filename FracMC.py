import cv2
import csv
import sys
import re
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

file = sys.argv[1]
nmues = int(sys.argv[2])

print(file)

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

def elipse(x,y,a,b,ang,i,N):
   fig = plt.figure(figsize=(10,10),dpi=102.4)
   ax = fig.add_subplot(111)
   ax.set_xlim(0,1)
   ax.set_ylim(0,1)
   for j in range(N):
      j=np.int64(j)
      ax.add_artist(Ellipse(xy=(x[j],y[j]), width=a[j], height=b[j], angle= ang[j],facecolor="black"))
   plt.axis('off')   
   fig.tight_layout()

   fig.canvas.draw()
   img = np.array(fig.canvas.renderer.buffer_rgba())

   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   ret, bin = cv2.threshold(gray,127,255,1)

   plt.clf()
   plt.close(fig)

   return(bin)

f = open('datos/DFS.csv', 'a', newline='')
writer = csv.writer(f)
for i in range(nstates):
   xi,yi,ai,bi,angi=x[i*N:(i+1)*N],y[i*N:(i+1)*N],a[i*N:(i+1)*N],b[i*N:(i+1)*N],ang[i*N:(i+1)*N]
   writer.writerow([i+500*nmues,fracpygen(elipse(xi,yi,ai,bi,angi,i,N))])
f.close()