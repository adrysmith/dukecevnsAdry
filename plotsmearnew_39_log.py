from numpy import loadtxt,log10,array,zeros,linspace,sin,cos,vectorize,arange,shape
from pylab import plot,show,xlabel,ylabel,title,legend,xlim,ylim,xscale,yscale,polyfit,contourf,axes,figure
from math import sqrt,log,exp,pi,inf,isclose,factorial,log10
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib.pyplot import text
import time as TIME
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import scipy

#whichNumFiles = 100 #how many files go into transfer matrix (with 10 sources per file)

print("Starting plotsmearnew...\n")

maxNph_prod = 13988 #manually assigned in this file...
narrmax = maxNph_prod # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
narrstep = 20 #the step for Ar39 is 20 (instead of 10) to make faster
narr = arange(0,narrmax,narrstep)

#xnew = arange(min(narr),max(narr),1)
#ynew = xnew
xnew = np.logspace(log10(1),log10(max(narr)),num=1000,base=10)
xnew = np.array([0] + xnew.tolist()) #adding 0 photons to start of list
ynew = xnew

print("maxNph_prod: ",maxNph_prod, "  narrstep: ",narrstep, "   min(xnew): ",min(xnew),"  max(xnew): ",max(xnew))

smearnewww = loadtxt("smearnew_39_log.txt")

print("Loaded: smearnew_39_log.txt")

plt.title("INTERP: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnewww,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew))
plt.ylim(0,max(xnew))
plt.show()

title("Slice at "+str(xnew[500]))
plot(smearnewww.T[500],'.')
show()

