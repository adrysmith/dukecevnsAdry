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

print("===================================================================")
thenScript = TIME.time()

numFiles = 1337 

###################
### FLUX ###
###################

tArr = loadtxt("tArr_flux_per_tSNburst_CEvNS.txt")#the time arr for CEvNS and Ar39 are same
photonsArr = loadtxt("photonsArr_flux_per_tSNburst_39.txt")
flux = loadtxt("flux_per_tSNburst_39_SHAPE_299_300.txt")
photonsArrInt = photonsArr.astype(int)


print(np.shape(tArr),np.shape(photonsArrInt),np.shape(flux.T))

f = scipy.interpolate.interp2d(tArr,photonsArrInt,flux.T,kind='cubic')
photonsArrIntnew = np.arange(min(photonsArrInt),max(photonsArrInt),1)
fluxnew = f(tArr,photonsArrIntnew)
#print("fluxnew: ",np.shape(fluxnew))

'''
title("INTERP: Distribution of $Photons_{detected}$ Flux (CEvNS)")
ylabel("Photons")
xlabel("time (ns)")
X,Y = np.meshgrid(tArr,photonsArrIntnew)
plt.pcolormesh(X,Y,fluxnew)
plt.colorbar(label="Photons/ns")
xscale('log')
yscale('symlog')
show() 
'''

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
print("maxNph_prod: ",maxNph_prod)
#smearnew = smearnew[0:maxNph_prod]

#smearexact = []
#for i in range(len(smearnew)):
#    smearexact.append(smearnew[i][0:maxNph_prod])

thenFiles = TIME.time()
counter = 0
#smeartotal_unnorm = np.zeros((700,maxNph_prod)) #unnormalized
path = "./smearmatrix13980/smearmatrix*"
print("Collecting:  ./smearmatrix13980/smearmatrix* files")
for filename in glob.glob(path):
	counter+=1
print("Total # files: ",counter)
print("Number of matrices used: ",numFiles)
counter = 0
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        temp = loadtxt(filename)
        temp = np.array(temp[0:maxNph_prod])
        smearexact = []
        for i in range(len(temp)):
            smearexact.append(temp[i][0:maxNph_prod]) #exact shape for Nph_det
        #print("np.shape(smearexact): ",np.shape(smearexact))
        if counter == 0: smeartotal_unnorm = np.array(smearexact)
        else: smeartotal_unnorm += np.array(smearexact)#updating the matrix, not growing it
	#smeartotal_unnorm.append(np.array(smearexact))
    print(counter)
    counter += 1
    if counter >=numFiles:
        break	
nowFiles = TIME.time()
print("Number of matrices: ",counter)
print("Duration for getting files: ",round(nowFiles-thenFiles,2)," sec\n")
#smeartotal_unnorm = np.array(smeartotal_unnorm)
#smeartotal = sum(smeartotal_unnorm) #add the individual matrices into one big transfer matrix

#for i in range(len(smeartotal)): #normalize that matrix by vertical column (i.e. per Nph_prod slice)
#    smeartotal[i] = smeartotal[i] / sum(smeartotal[i])

smeartotal = []
for i in range(len(smeartotal_unnorm)): #normalize that matrix by vertical column (ie per Nph_prod slice)
    smeartotal.append(smeartotal_unnorm[i] / sum(smeartotal_unnorm[i]))
smeartotal = np.array(smeartotal)

print("np.shape(smeartotal): ",np.shape(smeartotal))

narrmax = maxNph_prod # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
narrmax = 13980+1 #need to get 700 bins in the Nph_prod axis to match smeartotal, and with steps of 20, this narrmax gets us there
print("maxNph_prod: ",maxNph_prod)
narrstep = 20 #the step for Ar39 is 20 (instead of 10) to make faster
narr = arange(0,narrmax,narrstep)
ynarr = arange(0,maxNph_prod, 1)
#print("max(narr): ",max(narr))
#print("So... if use this maxnarr as arange: -->", np.shape(arange(0,max(narr),1)))

#print("smeartotal: ",np.shape(smeartotal),  " narr: ",np.shape(narr), " arange: ",np.shape(arange(0,max(narr),1)))
print("smeartotal: ",np.shape(smeartotal),  " narr: ",np.shape(narr), " arange: ",np.shape(ynarr))

print("\nInterpolating smeartotal...\n")
thenInterp = TIME.time()
f = scipy.interpolate.interp2d(narr,ynarr,smeartotal.T,kind='cubic')# Look! This is a catered "y" array
nowInterp = TIME.time()
print("Duration for interp2d: ",nowInterp-thenInterp, " sec")

xnew = arange(min(narr),max(narr),1)
ynew = arange(min(narr),max(narr),1)
#smearnew = f(xnew,ynew)
print("\n~~Desired dimensions of smearnew: ",np.shape(xnew)," (<< xnew full)")
xnew1 = arange(min(narr),int(max(narr)*(1/3)),1)
xnew2 = arange(max(xnew1)+1,int(max(narr)*(2/3)),1)
xnew3 = arange(max(xnew2)+1,max(narr),1)
ynew1,ynew2,ynew3 = xnew,xnew,xnew

thenSmearnew = TIME.time()
smearnew1 = f(xnew1,ynew1)
smearnew2 = f(xnew2,ynew2)
smearnew3 = f(xnew3,ynew3)
nowSmearnew = TIME.time()
print("Duration to make smearnew's: ",nowSmearnew-thenSmearnew," sec")

print("\nInterpolated")

print("smearnew1, 2, 3: ",np.shape(smearnew1),np.shape(smearnew2),np.shape(smearnew3))

smearnew = []
for partialmatrix in [smearnew1.T,smearnew2.T,smearnew3.T]:
	smearnew += list(partialmatrix)
print("smearnew: ",np.shape(smearnew)) 

thenWrite = TIME.time()

outfile=open("smearnew_39_Files_"+str(numFiles)+".txt",'w')
print("Writing to smearnew_39_Files_"+str(numFiles)+".txt")

for i in smearnew:
    for j in i:
        outfile.write(str(j)+ "  ")
    outfile.write("\n\n")
outfile.close()

nowWrite = TIME.time()
print("Duration for writing: ",nowWrite - thenWrite)
print("Written")

#plt.title("SUM: Smeared Nph_det vs. Nph_prod")
#X,Y = np.meshgrid(narr,arange(0,max(narr),1))
#plt.pcolormesh(X,Y,smeartotal.T,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
#plt.colorbar(label="probability")
#plt.xlabel("Nph_produced")
#plt.ylabel("Nph_detected")
#plt.xlim(0,1125)
#plt.ylim(0,1125)
#plt.show()
'''
print("Making interp plot")

plt.title("INTERP: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew))
plt.ylim(0,max(xnew))
print("Presumably, this is in memory....")
#plt.savefig("smearmatrix39.jpg",quality = 10)
#plt.show()
plt.close()    
    
print("~~~ Files (used in matrix) count: ",counter,' ~~~~\n\n')
'''


#Gauging the smoothness
'''
title("Slice at Nph_prod = 200")
plot(smearnew.T[200],'.')
xlim(0,200)
show()

title("Slice at Nph_prod = 800")
plot(smearnew.T[800],'.')
xlim(0,800)
show()

title("Slice at Nph_prod = 1200")
plot(smearnew.T[1200],'.')
xlim(0,1200)
show()

title("Slice at Nph_prod = "+str(maxNph_prod))
plot(smearnew.T[maxNph_prod-100],".")
xlim(0,maxNph_prod-100)
show()
'''
'''
smearnewww = loadtxt("smearnew_39.txt")
plt.title("INTERP: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnewww,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew))
plt.ylim(0,max(xnew))
plt.show()
'''
nowScript = TIME.time()
print("Duration of script: ",nowScript-thenScript," sec")
print("\nDone\n")
