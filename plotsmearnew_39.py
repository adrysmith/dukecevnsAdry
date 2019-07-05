from numpy import loadtxt,log10,array,zeros,linspace,sin,cos,vectorize,arange,shape
from pylab import plot,show,xlabel,ylabel,title,legend,xlim,ylim,xscale,yscale,polyfit,contourf,axes,figure
from math import sqrt,log,exp,pi,inf,isclose,factorial
import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib.pyplot import text
import time as TIME
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import scipy

whichNumFiles = 1337 #indicates which transfer matrix to multiply by (20, 100, etc files * 10 sources per file?)

print("Starting plotsmearnew...\n")

maxNph_prod = 13988 #manually assigned in this file...
narrmax = maxNph_prod # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
narrstep = 20 #the step for Ar39 is 20 (instead of 10) to make faster
narr = arange(0,narrmax,narrstep)

xnew = arange(min(narr),max(narr),1)
ynew = xnew

#print("maxNph_prod: ",maxNph_prod, "  narrstep: ",narrstep, "   min(xnew): ",min(xnew),"  max(xnew): ",max(xnew))

thenLoad = TIME.time()
print("Loading smearnew_39_Files_"+str(whichNumFiles)+".txt")
smearnew = loadtxt("smearnew_39_Files_"+str(whichNumFiles)+".txt")
nowLoad = TIME.time()
print("Loaded: smearnew_39.txt, duration: ",nowLoad - thenLoad, " sec")
print("\nsmearnew: ",np.shape(smearnew))

'''
plt.title("INTERP: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew))
plt.ylim(0,max(xnew))
plt.show()
'''

###################
### FLUX ###
###################

tArr = loadtxt("tArr_flux_per_tSNburst_CEvNS.txt")#the time arr for CEvNS and Ar39 are same
photonsArr = loadtxt("photonsArr_flux_per_tSNburst_39.txt")
flux = loadtxt("flux_per_tSNburst_39_SHAPE_299_300.txt")
photonsArrInt = photonsArr.astype(int)


print(np.shape(tArr),np.shape(photonsArrInt),np.shape(flux.T))

#####CHANGE THIS INTERP2D to 'linear' and NOT 'cubic'!!!!!!!!!!!!!!!!!!!1

f = scipy.interpolate.interp2d(tArr,photonsArrInt,flux.T,kind='cubic')
photonsArrIntnew = np.arange(min(photonsArrInt),max(photonsArrInt),1)
fluxnew = f(tArr,photonsArrIntnew)
fluxnewVmax = max(fluxnew.flatten())
print("fluxnew: ",np.shape(fluxnew))
print("Max fluxnew: ",fluxnewVmax)
print("min(photonsArrIntnew), and max(): ",min(photonsArrIntnew), max(photonsArrIntnew))


title("Distribution of $Photons_{produced}$ Flux (CEvNS)")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxnew)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()


#buffering the bottom of the photons and flux range so that Nph extends down from 2423 photons to 0.
fluxnew = list(fluxnew)
for i in range(min(photonsArrIntnew)):
	fluxnew.insert(0,np.zeros(299).tolist())
fluxnew = np.array(fluxnew[0:13980])#trim to accomodate transfer matrix
print("fluxnew after addition: ",np.shape(fluxnew))
 
photonsArrIntnew = np.arange(0,min(photonsArrIntnew),1).tolist() + photonsArrIntnew.tolist()
photonsArrIntnew = photonsArrIntnew[0:13980]
print("photonsArrIntnew after addition: ",np.shape(photonsArrIntnew),"   min/max: ",min(photonsArrIntnew), max(photonsArrIntnew))


print('====================================================')

maxPhotonDim = len(photonsArrIntnew) #13256 should be in here, due to using the Ar-39 count/keV/s plot which extends down to Nph=0

fluxsmeared = np.matmul(smearnew[0:maxPhotonDim,0:maxPhotonDim],fluxnew)
print("detected: ",np.shape(fluxsmeared))

title("Distribution of $Photons_{detected}$ Flux (CEvNS), auto max")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxsmeared)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()

title("Distribution of $Photons_{detected}$ Flux (CEvNS), vmax=0.16")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxsmeared,vmax=0.16)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()

title("Distribution of $Photons_{detected}$ Flux (CEvNS), vmax=fluxnewVmax")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxsmeared,vmax=fluxnewVmax)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()

title("Zoomed: Distribution of $Photons_{detected}$ Flux (CEvNS)")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew[2423:])
plt.pcolormesh(X,Y,fluxsmeared[2423:],vmax=0.16)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()


title("Slice at Nph_prod = 13000")
plot(smearnew[13000],'.')
xlim(0,13000)
show()
