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

maxFiles = 1950 #you determine this, with no limitations

print("\nStarting...")

###################
### FLUX ###
###################

tArr = loadtxt("tArr_flux_per_tSNburst_CEvNS.txt")
photonsArr = loadtxt("photonsArr_flux_per_tSNburst_CEvNS.txt")
flux = loadtxt("flux_per_tSNburst_CEvNS_SHAPE_299_470.txt")
photonsArrInt = photonsArr.astype(int)
print("photonsArrInt min/max: ",min(photonsArrInt),max(photonsArrInt))

print(np.shape(tArr),np.shape(photonsArrInt),np.shape(flux.T))

f = scipy.interpolate.interp2d(tArr,photonsArrInt,flux.T,kind='cubic')
photonsArrIntnew = np.arange(min(photonsArrInt),max(photonsArrInt),1)
fluxnew = f(tArr,photonsArrIntnew)
fluxnewVmax = max(fluxnew.flatten())
#print("fluxnew: ",np.shape(fluxnew))

title("INTERP: Distribution of $Photons_{detected}$ Flux (CEvNS)")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxnew)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show() 

###################
### SMEAR ###
###################
'''
smear1 = np.array(loadtxt("smearedprobs_normalized_reshape_500_4990_TIME_1549657086.392235.txt"))

narrmax = 5000 # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
narrstep = 10
narr = arange(0,narrmax,narrstep) #Nph_produced array #was 1 spacing

f = scipy.interpolate.interp2d(narr,arange(min(narr),max(narr),1),smear1.T,kind='cubic')
xnew = arange(min(narr),max(narr),1)
ynew = arange(min(narr),max(narr),1)
smearnew = f(xnew,ynew)

plt.title("INTERP:Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.show()

#Matrix multiplication: (n,k),(k,m)->(n,m)
#smearnew = (4990,4990) , fluxnew = (1125,299),  ->want: (1125,299)
#so need smearexact = (1125,1125)

maxNph_prod = max(photonsArrIntnew)+1#np.shape(fluxnew)[0] #if this doesn't seem right, use the other index ([1])
smearnew = smearnew[0:maxNph_prod]

smearexact = []
for i in range(len(smearnew)):
    smearexact.append(smearnew[i][0:maxNph_prod])


fluxsmeared = np.matmul(smearexact,fluxnew)
print("\n b dim: ",np.shape(b))

title("Smeared Distribution of $Photons_{detected/smeared}$ Flux (CEvNS)")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxsmeared)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show()  
'''



### PRINTING STAGE AND SMEARING STAGE ###

import glob

maxNph_prod = max(photonsArrIntnew)+1#np.shape(fluxnew)[0] #if this doesn't seem right, use the other index ([1])
#smearnew = smearnew[0:maxNph_prod]

#smearexact = []
#for i in range(len(smearnew)):
#    smearexact.append(smearnew[i][0:maxNph_prod])

counter = 0
#smeartotal_unnorm = [] #unnormalized, updated by adding individual matrices into one big transfer matrix
smeartotal_unnorm = np.zeros((150,maxNph_prod)) #these will be dimensions of smearexact, which we'll be adding with

path = "./smearmatrix1490/smearmatrix*"
print("Collecting:  ./smearmatrix1490/smearmatrix* files")
for filename in glob.glob(path):
        counter+=1
print("Total # files: ",counter)
print("Number of matrices used: ",maxFiles)
counter = 0 
thenadd = TIME.time()
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        counter +=1
        #print(filename)
        temp = loadtxt(filename)
        temp = np.array(temp[0:maxNph_prod])
        #print("temp: ",type(temp), " temp[0]: ",type(temp[0]))#
        print(counter)
        smearexact = []
        #print("len temp: ",len(temp), "   maxnph_prod: ",maxNph_prod)
        for i in range(len(temp)):
            smearexact.append(temp[i][0:maxNph_prod]) #exact shape for Nph_det
#        smeartotal_unnorm.append(np.array(smearexact))
        smeartotal_unnorm = np.array(smeartotal_unnorm) + np.array(smearexact) #for sake of space, updating array, not adding to it
    if counter > maxFiles: break

nowadd = TIME.time()
print("Duration of adding files: ",round(nowadd-thenadd,2)," sec")
#smeartotal_unnorm = np.array(smeartotal_unnorm)
#smeartotal = sum(smeartotal_unnorm) #add the individual matrices into one big transfer matrix
smeartotal = np.array(smeartotal_unnorm)
for i in range(len(smeartotal)): #normalize that matrix by vertical column (i.e. per Nph_prod slice)
    smeartotal[i] = smeartotal[i] / sum(smeartotal[i])

narrmax = 1500 # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
narrstep = 10
narr = arange(0,narrmax,narrstep)
ynarr = arange(0,maxNph_prod,1)
print("maxNph_prod: ",maxNph_prod)
print("smeartotal: ",np.shape(smeartotal),  " narr: ",np.shape(narr), " arange: ",np.shape(arange(0,maxNph_prod,1)))

f = scipy.interpolate.interp2d(narr,ynarr,smeartotal.T,kind='linear')
xnew = arange(min(narr),max(narr),1)
ynew = arange(min(narr),max(narr),1)
smearnew = f(xnew,ynew) #smearnew has 1x1 bins
print("smearnew[0][0]: ",smearnew[0][0])

smear1 = smearnew
smear1count = []
for slice in smear1.T:
	smear1count.append(sum(slice))

smearnorm = []
for slice in smearnew.T:
	smearnorm.append(slice / sum(slice))
print("smearnorm: ",np.shape(smearnorm))
smearnew = np.array(smearnorm).T
smear2 = smearnew
smear2count = []
for slice in smear2.T:
	smear2count.append(sum(slice))
'''
plt.title("Comparing Interp2D Transfer Matrix w/(o) Extra Normalization")
plt.plot(smear1count,label="No extra normalization")
plt.plot(smear2count,label="Re-normalized")
plt.legend()
plt.show()
'''


plt.title("SUM: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(narr,arange(0,maxNph_prod,1))
plt.pcolormesh(X,Y,smeartotal.T,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,1125)
plt.ylim(0,1125)
plt.show()



plt.title("INTERP: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,1125)
plt.ylim(0,1125)
plt.show()

'''    
plt.title("INTERP: Smeared Nph_det vs. Nph_prod (Zoomed in)")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,150)
plt.ylim(0,150)
plt.show()
'''
    
#print("~~~ File count: ",counter,' ~~~~')



#Gauging the smoothness
'''
title("Slice at Nph_prod = 0")
plot(smearnew.T[0],'.')
xlim(0,0+50)
show()

title("Slice at Nph_prod = 5")
plot(smearnew.T[5],'.')
xlim(0,5+50)
show()

title("Slice at Nph_prod = 20")
plot(smearnew.T[20],'.')
xlim(0,20+50)
show()

title("Slice at Nph_prod = 50")
plot(smearnew.T[50],'.')
xlim(0,50+50)
show()

title("Slice at Nph_prod = 100")
plot(smearnew.T[100],'.')
xlim(0,100+50)
show()
title("Slice at Nph_prod = 1200")
plot(smearnew.T[1200],'.')
xlim(0,1200+20)
show()
'''

