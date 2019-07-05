'''in this file, I apply transfer matrix to 39 and CEvNS'''

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

boolPlotPreSmear = 0
boolPlotPostSmear = 0
boolPlotPostSmearCombo = 0
boolPlotResidual = 1
boolPlotSigma = 1
boolRate = 1
boolCounts = 0

whichlowNphFiles = 740 #how many files go into the low Nph range transfer matrix (corresponds to existing file)
whichNtons = 2500 # 1/4 of 10-kT module

thenAll = TIME.time()
###################################
####### C E v N S  ###############
##################################

valuesPhotonsPlot = loadtxt('valuesPhotonsPlot_CEvNS_SHAPE_299_470_Ntons_'+str(whichNtons)+'.txt') # np.shape = (299,470)
photons = loadtxt('photonsArr_flux_per_tSNburst_CEvNS.txt') # np.shape = 470
time = loadtxt('tArr_flux_per_tSNburst_CEvNS.txt') # np.shape = 299, same time for CEvns and Ar39
time /= (10**9) #puts time into sec (instead of ns)

photons_1x1 = np.arange(min(photons),max(photons),1)
print(np.shape(photons_1x1))
f_CEvNS = scipy.interpolate.interp2d(time,photons,valuesPhotonsPlot.T)

valuesPhotonsPlot_1x1 = f_CEvNS(time,photons_1x1) #by '1x1' I just mean 1-photon binning, but log time per usual

if boolPlotPreSmear:
	plt.title("CEvNS Photons vs. Time (1x1) (Pre-Smear) \n(lowNph)")
	X,Y = np.meshgrid(time,photons_1x1)
	plt.pcolormesh(X,Y, valuesPhotonsPlot_1x1)
	plt.colorbar(label="events/Nph/sec")
	plt.xlabel("time (s)")
	plt.ylabel("photons")
	plt.xscale('log')
	#plt.ylim(0,200)
	plt.show()


#################################
########  A R  3 9 #############
#################################

valuesPhotonsPlot39 = loadtxt('valuesPhotonsPlot_39_SHAPE_299_300_Ntons_'+str(whichNtons)+'.txt') # np.shape = (299,300)
photons39 = loadtxt('photonsArr_flux_per_tSNburst_39.txt') #np.shape = 300

photons39_1x1 = np.arange(min(photons39),max(photons39),1)
print('photons39_1x1: ',np.shape(photons39_1x1))
f_39 = scipy.interpolate.interp2d(time,photons39,valuesPhotonsPlot39.T)
valuesPhotonsPlot39_1x1 = f_39(time,photons39_1x1) # by '1x1 I just mean 1-photon binning, but log time as usual

if boolPlotPreSmear:
	plt.title("Ar39 Photons vs. Time (1x1) (Pre-Smear)")
	X,Y = np.meshgrid(time,photons39_1x1)
	plt.pcolormesh(X,Y, valuesPhotonsPlot39_1x1)
	plt.colorbar(label="events/Nph/sec")
	plt.xlabel("time (s)")
	plt.ylabel("photons")
	plt.xscale('log')
	plt.ylim(0,200)
	plt.show()


###################################
######  Load Transfer Matrix  #####
###################################
print("max(photons39_1x1) ----> ",max(photons39_1x1))
whichNumFiles = 1337 #used to access the transfer matrix with numFiles incorporated (numFiles*10sources/file = sources used)

print("Loading smearnew_39_Files_"+str(whichNumFiles)+".txt")
thenLoad = TIME.time()
smearfull = loadtxt("smearnew_39_Files_"+str(whichNumFiles)+".txt") # this matrix is 1x1 binning, generated from Use_transfer_matrix...py 
nowLoad = TIME.time()
print("Loaded: smearnew_39_Files_"+str(whichNumFiles)+".txt, duration: ",round(nowLoad - thenLoad,2), " sec")

######################  A R  3 9 ##############################
#\\\\ Trim down for Ar39  multiplication /////
maxNph_prod_39 = int(max(photons39_1x1)+1) #max Ar39 photons
smear39 = np.array(smearfull)[:maxNph_prod_39,:maxNph_prod_39] #smear39 is smearfull, but tailored down to Ar39  # this matrix is 1x1 binning, generated from Use_transfer_matrix...pyrange
print("smear39: ",np.shape(smear39))
#\\\\  Patch over the lower Nph < 200 range (avoid streak) //////
smearLowNph = loadtxt("smearnew_lowNph_Files_"+str(whichlowNphFiles)+".txt") #comes from Diff_binomial_pythonsimplified_LowNph.ipynb on Macbook

smear39[:smearLowNph.shape[0],:smearLowNph.shape[1]] = smearLowNph #add in this lowNph range to try to fix 'streak'
for i in range(len(smear39)):
        for j in range(len(smear39[0])):
                if j>i: smear39[i][j] = 0 #unphysical values above the y=x line

def normalize_smear(smear):
	for i in range(len(smear)): 
        	smear[i] = smear[i]/sum(smear[i])
	return smear
smear39 = normalize_smear(smear39) #renormalize

print("smear39.T: ",np.shape(smear39), "  valuesPhotonsPlot39_1x1: ",np.shape(valuesPhotonsPlot39_1x1))
valuesPhotonsPlot39_1x1_smeared = np.matmul(smear39.T,valuesPhotonsPlot39_1x1)
print("valuesPhotonsPlot39_1x1_smeared: ",np.shape(valuesPhotonsPlot39_1x1_smeared))

#\\\\\ Suppress negative #s and #s < 10^-10 /////////
for i in range(len(valuesPhotonsPlot39_1x1_smeared)):
	for j in range(len(valuesPhotonsPlot39_1x1_smeared[0])):
		if valuesPhotonsPlot39_1x1_smeared[i][j] < 0 or valuesPhotonsPlot39_1x1_smeared[i][j] < (10**-10):
			valuesPhotonsPlot39_1x1_smeared[i][j] = 0


if boolPlotPostSmear:
        plt.title("Ar39 Photons vs. Time (1x1) (Smeared)")
        X,Y = np.meshgrid(time,photons39_1x1)
        plt.pcolormesh(X,Y, valuesPhotonsPlot39_1x1_smeared)
        plt.colorbar(label="events/Nph/sec")
        plt.xlabel("time (s)")
        plt.ylabel("photons")
        plt.xscale('log')
        #plt.ylim(0,1000)
        plt.show()

##################### C E v N S ########################
#\\\\ Trim down for CEvNS  multiplication /////
maxNph_prod_CEvNS = int(max(photons_1x1)+1) #max CEvNS  photons
print("maxNphcevns: ",np.shape(maxNph_prod_CEvNS))
smearCEvNS = np.array(smearfull)[:maxNph_prod_CEvNS,:maxNph_prod_CEvNS] #smearCEvNS is smearfull, but tailored down to cevns  # this matrix is 1x1 binning, generated from Use_transfer_matrix...pyrange
print("smearCEvNS: ",np.shape(smearCEvNS))
#\\\\  Patch over the lower Nph < 200 range (avoid streak) //////
smearLowNph = loadtxt("smearnew_lowNph_Files_"+str(whichlowNphFiles)+".txt") #comes from Diff_binomial_pythonsimplified_LowNph.ipynb on Macbook

smearCEvNS[:smearLowNph.shape[0],:smearLowNph.shape[1]] = smearLowNph #add in this lowNph range to try to fix 'streak'
for i in range(len(smearCEvNS)):
        for j in range(len(smearCEvNS[0])):
                if j>i: smearCEvNS[i][j] = 0 #unphysical values above the y=x line

smearCEvNS = normalize_smear(smearCEvNS) #renormalize

print("smearCEvNS.T: ",np.shape(smearCEvNS), "  valuesPhotonsPlot_1x1: ",np.shape(valuesPhotonsPlot_1x1))
valuesPhotonsPlot_1x1_smeared = np.matmul(smearCEvNS.T,valuesPhotonsPlot_1x1)
print("valuesPhotonsPlot_1x1_smeared: ",np.shape(valuesPhotonsPlot_1x1_smeared))

#\\\\\ Suppress negative #s and #s < 10^-10 /////////
for i in range(len(valuesPhotonsPlot_1x1_smeared)):
        for j in range(len(valuesPhotonsPlot_1x1_smeared[0])):
                if valuesPhotonsPlot_1x1_smeared[i][j] < 0 or valuesPhotonsPlot_1x1_smeared[i][j] < (10**-10):
                        valuesPhotonsPlot_1x1_smeared[i][j] = 0

if boolPlotPostSmear:
        plt.title("CEvNS Photons vs. Time (1x1) (Smeared)")
        X,Y = np.meshgrid(time,photons_1x1)
        plt.pcolormesh(X,Y, valuesPhotonsPlot_1x1_smeared)
        plt.colorbar(label="events/Nph/sec")
        plt.xlabel("time (s)")
        plt.ylabel("photons")
        plt.xscale('log')
        #plt.ylim(0,1000)
        plt.show()


#############################################################################
###############  R E S I D U A L   P L O T  ################################
#############################################################################

################### Making Combo (Ar39+Cevns) Distribution ##################
print("making combo distr...")
valuesPhotonsPlot_1x1_smeared_padded = np.zeros((valuesPhotonsPlot39_1x1_smeared.shape[0],valuesPhotonsPlot39_1x1_smeared.shape[1]))#make padded version of cevns distr
valuesPhotonsPlot_1x1_smeared_padded[:valuesPhotonsPlot_1x1_smeared.shape[0],:valuesPhotonsPlot_1x1_smeared.shape[1]] = valuesPhotonsPlot_1x1_smeared
print('valuesPhotonsPlot_1x1_smeared_padded: ',np.shape(valuesPhotonsPlot_1x1_smeared_padded))

for i in range(len(valuesPhotonsPlot_1x1_smeared_padded)):
	for j in range(len(valuesPhotonsPlot_1x1_smeared_padded[0])):
		if valuesPhotonsPlot_1x1_smeared_padded[i][j] <0: 
                        print("valuesPhotonsPlot_1x1_smeared_padded[",i,"][",j,"] = ",valuesPhotonsPlot_1x1_smeared_padded[i][j])
                        valuesPhotonsPlot_1x1_smeared_padded[i][j] = 0 
		if valuesPhotonsPlot39_1x1_smeared[i][j] <0: 
                        print("valuesPhotonsPlot39_1x1_smeared[",i,"][",j,"] = ",valuesPhotonsPlot39_1x1_smeared[i][j])
                        valuesPhotonsPlot39_1x1_smeared[i][j] = 0 


valuesPhotonsPlotCombo_1x1_smeared = valuesPhotonsPlot39_1x1_smeared + valuesPhotonsPlot_1x1_smeared_padded
print('valuesPhotonsPlotCombo_1x1_smeared: ',np.shape(valuesPhotonsPlotCombo_1x1_smeared))

if boolPlotPostSmearCombo:
        plt.title("CEvNS+Ar39 (Combo)  Photons vs. Time (1x1) (Smeared)")
        X,Y = np.meshgrid(time,photons39_1x1)
        plt.pcolormesh(X,Y, valuesPhotonsPlotCombo_1x1_smeared)
        plt.colorbar(label="events/Nph/sec")
        plt.xlabel("time (s)")
        plt.ylabel("photons")
        plt.xscale('log')
        plt.show()


############# Calculating residual matrix #####################
print("residual plot...")

residualPlot_rate = valuesPhotonsPlot_1x1_smeared_padded - valuesPhotonsPlot39_1x1_smeared

if boolPlotResidual:
        if boolRate:
                plt.clf()
                plt.title("Residual (RATE) in Photons vs. Time (1x1) (Smeared) (SymLogNorm) \n(CEvNS - Ar39)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_rate, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5)))
                plt.colorbar(label="events/Nph/sec")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (RATE) in Photons vs. Time (1x1) (Smeared) (Linear) \n(CEvNS - Ar39)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_rate, cmap='bwr')
                plt.colorbar(label="events/Nph/sec")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

############### Calculating Sigma matrix  #####################
# sigma for each bin = Signal / sqrt(signal +background)
print("sigma plot...")
sigmaPlot_rate = valuesPhotonsPlot_1x1_smeared_padded

def calc_sigmaPlot(sigmaPlot,valuesPhotonsPlot_1x1_smeared_padded,valuesPhotonsPlot39_1x1_smeared):
	for i in range(len(sigmaPlot)):
		for j in range(len(sigmaPlot[0])):
			if valuesPhotonsPlot_1x1_smeared_padded[i][j] <0:
				print("valuesPhotonsPlot_1x1_smeared_padded[",i,"][",j,"] = ",valuesPhotonsPlot_1x1_smeared_padded[i][j])
				valuesPhotonsPlot_1x1_smeared_padded[i][j] = 0
			if valuesPhotonsPlot39_1x1_smeared[i][j] <0:
				print("valuesPhotonsPlot39_1x1_smeared[",i,"][",j,"] = ",valuesPhotonsPlot39_1x1_smeared[i][j])
				valuesPhotonsPlot39_1x1_smeared[i][j] = 0
			if sqrt(valuesPhotonsPlot_1x1_smeared_padded[i][j] + valuesPhotonsPlot39_1x1_smeared[i][j]) >0:
				sigmaPlot[i][j] /= sqrt(valuesPhotonsPlot_1x1_smeared_padded[i][j] + valuesPhotonsPlot39_1x1_smeared[i][j])
			else: sigmaPlot[i][j] = 0
	return sigmaPlot

sigmaPlot_rate = calc_sigmaPlot(sigmaPlot_rate,valuesPhotonsPlot_1x1_smeared_padded,valuesPhotonsPlot39_1x1_smeared)

if boolPlotSigma:
        if boolRate:
                plt.clf()
                plt.title("Sigma ($\sigma$) (RATE) in Photons vs. Time (1x1) \n$\\frac{s}{\sqrt{s+b}}$")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, sigmaPlot_rate)
                plt.colorbar(label="$ \\frac{s}{\sqrt{s+b}}$")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

#add sigmas of each bin in quadrature
def calc_deltaChiSquared(sigmaPlot):
	deltaChiSquared = 0
	for i in sigmaPlot.flatten():
		deltaChiSquared += i
	deltaChiSquared = sqrt(deltaChiSquared)
	print('deltaChiSquared: ',deltaChiSquared)

print("From rate sigma: ",calc_deltaChiSquared(sigmaPlot_rate))
nowAll = TIME.time()
print("DONE... Duration: ",round(nowAll-thenAll,2),' sec')
	
