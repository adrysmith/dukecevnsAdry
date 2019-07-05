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

whichNumFiles = 1337 #used to access the transfer matrix with numFiles incorporated (numFiles*10sources/file = sources used)

print("Starting plotsmearnew...\n")

############################
######  FLUX  #######
############################
tArr = loadtxt("tArr_flux_per_tSNburst_CEvNS.txt")#the time arr for CEvNS and Ar39 are same
photonsArr = loadtxt("photonsArr_flux_per_tSNburst_CEvNS.txt")
flux = loadtxt("flux_per_tSNburst_CEvNS_SHAPE_299_470.txt")
photonsArrInt = photonsArr.astype(int)


print(np.shape(tArr),np.shape(photonsArrInt),np.shape(flux.T))

f = scipy.interpolate.interp2d(tArr,photonsArrInt,flux.T,kind='cubic')
photonsArrIntnew = np.arange(min(photonsArrInt),max(photonsArrInt),1)
fluxnew = f(tArr,photonsArrIntnew)
fluxnewVmax = max(fluxnew.flatten())
print("fluxnew: ",np.shape(fluxnew))
print("Max fluxnew: ",fluxnewVmax)
print("min(photonsArrIntnew), and max(): ",min(photonsArrIntnew), max(photonsArrIntnew))
print("len(photonsArrIntnew): ",len(photonsArrIntnew))

title("Distribution of $Photons_{produced}$ Flux (CEvNS)")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxnew)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()

#############################
#############################

thenLoad = TIME.time()
print("Loading smearnew_39_Files_"+str(whichNumFiles)+".txt")
smearnew = loadtxt("smearnew_39_Files_"+str(whichNumFiles)+".txt") # this matrix is 1x1 binning, generated from Use_transfer_matrix...py 
nowLoad = TIME.time()
print("Loaded: smearnew_39_Files_"+str(whichNumFiles)+".txt, duration: ",nowLoad - thenLoad, " sec")
print("\nsmearnew: ",np.shape(smearnew))

#Plotting (log spacing) of transfer matrix -- code from Use_transfer_matrix_pythonsimplified_Ar39levels_LOG.py
print("Interpolating & Plotting the Transfer Matrix")
#theninterp = TIME.time()

maxNph_prod = max(photonsArrIntnew)+1 #max CEvNS photons
smeartotal = np.array(smearnew)[:maxNph_prod,:maxNph_prod] #smeartotal is smearnew, but tailored down to CEvNS  # this matrix is 1x1 binning, generated from Use_transfer_matrix...pyrange

smeartotalLowNph = loadtxt("smearnew_lowNph_Files_240.txt") #comes from Diff_binomial_pythonsimplified_LowNph.ipynb on Macbook

smeartotal[:smeartotalLowNph.shape[0],:smeartotalLowNph.shape[1]] = smeartotalLowNph #add in this lowNph range to try to fix 'streak'
for i in range(len(smeartotal)):
	for j in range(len(smeartotal[0])):
		if j>i: smeartotal[i][j] = 0 #unphysical values above the y=x line

for i in range(len(smeartotal)): #renormalize
	smeartotal[i] = smeartotal[i]/sum(smeartotal[i])


narr = np.arange(0,maxNph_prod,1) #goes up to max CEvNS photons
ynarr = narr
print("smeartotal (trimmed): ",np.shape(smeartotal),  " narr: ",np.shape(narr), " ynarr: ",np.shape(ynarr))

#f = scipy.interpolate.interp2d(narr,ynarr,smeartotal.T,kind='linear')
f = scipy.interpolate.RectBivariateSpline(narr,ynarr,smeartotal.T,kx=1,ky=1)
xnew_LOG = np.logspace(log10(1),log10(max(narr)),num=1000,base=10)
xnew_LOG  = np.array( [0] + xnew_LOG.tolist()) #adding 0 photons to start of list
ynew_LOG  = xnew_LOG
smearnew_LOG = f(xnew_LOG,ynew_LOG)

#LOG plotting
plt.title("INTERP: Smeared Nph_det vs. Nph_prod (trimmed to CEvNS range) (LOG)")
X,Y = np.meshgrid(xnew_LOG,ynew_LOG)
plt.pcolormesh(X,Y,smearnew_LOG,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew_LOG))
plt.ylim(0,max(ynew_LOG))
plt.show()

xnew_1x1 = arange(min(narr),max(narr),1)
ynew_1x1 = xnew_1x1
smearnew_1x1 = f(xnew_1x1,ynew_1x1) #1x1 bins

smearnorm_1x1 = []
for slice in smearnew_1x1.T:
        smearnorm_1x1.append(slice / sum(slice))
print("smearnorm_1x1: ",np.shape(smearnorm_1x1))
smearnew_1x1 = np.array(smearnorm_1x1).T


#nowinterp = TIME.time()
#print("Duration for interpolation: ",nowinterp - theninterp, " sec\n")

plt.title("INTERP: Smeared Nph_det vs. Nph_prod (trimmed to CEvNS range) (1x1bin)")
X,Y = np.meshgrid(xnew_1x1,ynew_1x1)
plt.pcolormesh(X,Y,smearnew_1x1,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew_1x1))
plt.ylim(0,max(ynew_1x1))
plt.show()

#Compare with above plot of 1x1 bins... I think smearnew_1x1 is redundant since smeartotal is already 1x1
xnewsmeartotal = arange(0,maxNph_prod,1)
ynewsmeartotal = xnewsmeartotal
print("xnewsmeartotal: ",np.shape(xnewsmeartotal),"  smeartotal: ",np.shape(smeartotal))

plt.title("NON-INTERP: Smeared Nph_det vs. Nph_prod (trimmed to CEvNS range) (from smeartotal trimmed)")
X,Y = np.meshgrid(xnewsmeartotal,ynewsmeartotal)
plt.pcolormesh(X,Y,smeartotal,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
#plt.xlim(0,)
#plt.ylim(0,max(ynew_1x1))
plt.show()

#End code from Use_transfer_matrix_pythonsimplified_Ar39levels_LOG.py
'''
plt.title("INTERP: #smeartotal is smearnew, but tailored down to CEvNS rang # this matrix is 1x1 binning, generated from Use_transfer_matrix...pye Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew))
plt.ylim(0,max(xnew))
plt.show()
'''

''' Moved this block to t #smeartotal is smearnew, but tailored down to CEvNS rang # this matrix is 1x1 binning, generated from Use_transfer_matrix...pyeop of file
###################
### FLUX ###
###################

tArr = loadtxt("tArr_flux_per_tSNburst_CEvNS.txt")#the time arr for CEvNS and Ar39 are same
photonsArr = loadtxt("photonsArr_flux_per_tSNburst_39.txt")
flux = loadtxt("flux_per_tSNburst_39_SHAPE_299_300.txt")
photonsArrInt = photonsArr.astype(int)


print(np.shape( #smeartotal is smearnew, but tailored down to CEvNS rangetArr),np.shape(photonsArrInt),np.shape(flux.T))

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
'''

#buffering the bottom of the photons and flux range so that Nph extends down from 2423 photons to 0.

'''
fluxnew = list(fluxnew)
for i in range(min(photonsArrIntnew)):
	fluxnew.insert(0,np.zeros(299).tolist())
fluxnew = np.array(fluxnew[0:13980])#trim to accomodate transfer matrix
print("fluxnew after addition: ",np.shape(fluxnew))
 
photonsArrIntnew = np.arange(0,min(photonsArrIntnew),1).tolist() + photonsArrIntnew.tolist()
photonsArrIntnew = photonsArrIntnew[0:13980]
print("photonsArrIntnew after addition: ",np.shape(photonsArrIntnew),"   min/max: ",min(photonsArrIntnew), max(photonsArrIntnew))
'''

print('====================================================')
maxPhotonDim = len(photonsArrIntnew) #13256 should be in here, due to using the Ar-39 count/keV/s plot which extends down to Nph=0

print("smearnew[0:maxPhotonDim,0:maxPhotonDim]: ",np.shape(smearnew[0:maxPhotonDim,0:maxPhotonDim]))
print("fluxnew: ",np.shape(fluxnew))

fluxsmeared = np.matmul(smearnew[0:maxPhotonDim,0:maxPhotonDim],fluxnew)
print("detected: ",np.shape(fluxsmeared))

title("Distribution of $Photons_{detected}$ Flux (CEvNS), auto max")
ylabel("Photons")
xlabel("time (ns) #smeartotal is smearnew, but tailored down to CEvNS range")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxsmeared)
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

#title("slice at Nph_prod = 10")
#plot(smearnew[10],'.')
#xlim(0,100)
#show()

#title("Slice at Nph_prod = 13000")
#plot(smearnew[13000],'.')
#xlim(0,13000)
#show()
