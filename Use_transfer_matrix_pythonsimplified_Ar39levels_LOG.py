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
import subprocess

boolPrintDistributionInitial = False
#maxFiles = 1337 #max index # of files to include in matrix
maxFiles = 10 #since this PY script actually generates the transfer matrix, this # can be anything (i.e. not a pre-existing "whichNumFiles" variable call for a transfer file which has already been generated)

print("Starting...")
thenscript = TIME.time()


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

if boolPrintDistributionInitial == True:
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
smeartotal_unnorm = [] #unnormalized

path = "./smearmatrix13980/smearmatrix*"
print("Collecting:  ./smearmatrix13980/smearmatrix* files")
for filename in glob.glob(path):
	counter+=1
print("Total # files: ",counter)
print("Files used: ",maxFiles)
counter = 0
thencounter = TIME.time()
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        print(counter)
        counter +=1
        #print(filename)
        temp = loadtxt(filename)
        temp = np.array(temp[0:maxNph_prod])
        #print("temp: ",type(temp), " temp[0]: ",type(temp[0]))#
        smearexact = []
#CHANGE THIS... NEED TO UPDATE MATRIX, NOT APPEND AND GROW
        for i in range(len(temp)):
            smearexact.append(temp[i][0:maxNph_prod]) #exact shape for Nph_det
        smeartotal_unnorm.append(np.array(smearexact))
    if counter >maxFiles:
        break	
nowcounter = TIME.time()
#print("Number of matrices in total matrix: ",counter)
print("Duration for adding: ",nowcounter-thencounter," sec\n")

smeartotal_unnorm = np.array(smeartotal_unnorm)
smeartotal = sum(smeartotal_unnorm) #add the individual matrices into one big transfer matrix
for i in range(len(smeartotal)): #normalize that matrix by vertical column (i.e. per Nph_prod slice)
    smeartotal[i] = smeartotal[i] / sum(smeartotal[i])

narrmax = maxNph_prod # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
narrmax = 13980+1 #need 700 bins in Nph_prod
print("maxNph_prod: ",maxNph_prod)
narrstep = 20 #the step for Ar39 is 20 (instead of 10) to make faster
narr = arange(0,narrmax,narrstep)
ynarr = arange(0,maxNph_prod,1)
#print("max(narr): ",max(narr))
#print("So... if use this maxnarr as arange: -->", np.shape(arange(0,max(narr),1)))

print("smeartotal: ",np.shape(smeartotal),  " narr: ",np.shape(narr), " arange: ",np.shape(ynarr))

print("\nInterpolating smeartotal...\n")
theninterp = TIME.time()
f = scipy.interpolate.interp2d(narr,ynarr,smeartotal.T,kind='cubic')# Look! This is a catered "y" array
xnew = np.logspace(log10(1),log10(max(narr)),num=1000,base=10)
xnew = np.array( [0] + xnew.tolist()) #adding 0 photons to start of list
ynew = xnew
smearnew = f(xnew,ynew)
nowinterp = TIME.time()
print("Duration for interpolation: ",nowinterp - theninterp, " sec\n")
#print("min(xnew): ",min(xnew),"  max(xnew): ",max(xnew),"  len(xnew): ",len(xnew))
#print("np.shape(smearnew): ",np.shape(smearnew))

#xnew = arange(min(narr),max(narr),1)
#ynew = arange(min(narr),max(narr),1)
#smearnew = f(xnew,ynew)


outfile=open("smearnew_39_log_Files_"+str(maxFiles)+".txt",'w')
print("Writing to smearnew_39_log_Files_"+str(maxFiles)+".txt")
thenwrite = TIME.time()
for i in smearnew:
    for j in i:
        outfile.write(str(j)+ "  ")
    outfile.write("\n\n")
outfile.close()
nowwrite = TIME.time()
print("Duration to write: ",nowwrite - thenwrite, " sec\n")
print("Interpolated & Written")

#plt.title("SUM: Smeared Nph_det vs. Nph_prod")
#X,Y = np.meshgrid(narr,arange(0,max(narr),1))
#plt.pcolormesh(X,Y,smeartotal.T,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
#plt.colorbar(label="probability")
#plt.xlabel("Nph_produced")
#plt.ylabel("Nph_detected")
#plt.xlim(0,1125)
#plt.ylim(0,1125)
#plt.show()

plt.title("INTERP: Smeared Nph_det vs. Nph_prod")
X,Y = np.meshgrid(xnew,ynew)
plt.pcolormesh(X,Y,smearnew,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
plt.colorbar(label="probability")
plt.xlabel("Nph_produced")
plt.ylabel("Nph_detected")
plt.xlim(0,max(xnew))
plt.ylim(0,max(xnew))
  
plt.savefig("smearmatrix39_log_Files_"+str(maxFiles)+".png")
plt.show() 
#subprocess.run(['ls'])    
#print("~~~ File count: ",counter,' ~~~~\n\n')


'''
#Gauging the smoothness
title("Slice at Nph_prod = 200")
plot(smearnew.T[200],'.')
#xlim(0,200)
show()

title("Slice at Nph_prod = 800")
plot(smearnew.T[800],'.')
#xlim(0,800)
show()

title("Slice at Nph_prod = 900")
plot(smearnew.T[900],'.')
#xlim(0,900)
show()

#title("Slice at Nph_prod = "+str(maxNph_prod))
#plot(smearnew.T[maxNph_prod-100],".")
#xlim(0,maxNph_prod-100)
#show()
'''
'''
smearnewww = loadtxt("smearnew_39_log.txt")
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
nowscript = TIME.time()
print("Total duration: ", nowscript-thenscript, " sec\n(DONE)\n")
