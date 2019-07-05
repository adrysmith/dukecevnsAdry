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
import pickle

boolPlotSmear39 = 0
boolPlotPreSmear = 0
boolPlotPostSmear = 0
boolPlotPostSmearCombo = 0
boolPlotRatio = 1
boolPlotResidual =0 
boolPlotSigma = 1
boolRate = 0
boolCounts = 1

whichlowNphFiles = 740 #how many files go into the low Nph range transfer matrix (corresponds to existing file)
whichNtons = 2500 # 1/4 of 10-kT module

var30 = 10 #how many time slices to make (linear)... was 30 --> resulted in ~0.3sec bin sizes

thenAll = TIME.time()
###################################
####### C E v N S  ###############
##################################
print("cevns...")
#instead of loadtxt, using pickle.load:
fileObject = open('valuesPhotonsPlot_CEvNS_SHAPE_299_470_Ntons_'+str(whichNtons)+'_pickle.txt','rb')
valuesPhotonsPlot = pickle.load(fileObject)
#valuesPhotonsPlot = loadtxt('valuesPhotonsPlot_CEvNS_SHAPE_299_470_Ntons_'+str(whichNtons)+'.txt') # np.shape = (299,470)
fileObject = open('photonsArr_flux_per_tSNburst_CEvNS_pickle.txt','rb')
photons = pickle.load(fileObject)
#photons = loadtxt('photonsArr_flux_per_tSNburst_CEvNS_pickle.txt') # np.shape = 470
fileObject = open('tArr_flux_per_tSNburst_CEvNS_pickle.txt','rb')
time = pickle.load(fileObject)
#time = loadtxt('tArr_flux_per_tSNburst_CEvNS_pickle.txt') # np.shape = 299, same time for CEvns and Ar39
time /= (10**9) #puts time into sec (instead of ns)

photons_1x1 = np.arange(min(photons),max(photons),1)
print(np.shape(photons_1x1))
f_CEvNS = scipy.interpolate.interp2d(time,photons,valuesPhotonsPlot.T)
print("done interpolating valuesPhotonsPlot")

valuesPhotonsPlot_1x1 = f_CEvNS(time,photons_1x1) #by '1x1' I just mean 1-photon binning, but log time per usual
time_30 = linspace(min(time),max(time),var30)
time_30_binsize = time_30[1]-time_30[0]
print("Time_30_binsize: ",time_30_binsize)

valuesPhotonsPlot_1x1_t30 = f_CEvNS(time_30,photons_1x1)

if boolPlotPreSmear:
	xlabels = []
	ylabels = []
	for i in range(len(time_30)):
		if not 30%5: xlabels.append(str(int(i)))
		else: xlabels.append('')
	for i in range(len(photons_1x1)):
		if not max(photons_1x1)%10: ylabels.append(str(int(i)))
		else: ylabels.append('')
	plt.title("CEvNS Photons vs. Time (Pre-Smear)")
	X,Y = np.meshgrid(time_30,photons_1x1)
	plt.pcolormesh(X,Y, valuesPhotonsPlot_1x1_t30)
	plt.colorbar(label="events/Nph/sec")
	plt.xlabel("time (s)")
	plt.ylabel("photons")
	#plt.xscale('log')
	#axes.Axes.set_axisbelow(True)
	#yaxis.grid(color='gray', linestyle='dashed')	
	#fig = plt.figure()
	#ax = fig.gca()
	ax = plt.gca()
	ax.set_xticks(time_30)
	#ax.set_yticks(photons_1x1)
	#ax.set_xticklabels(xlabels)
	#ax.set_yticklabels(ylabels)
	#ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	#ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.grid()
	ax.set_xticklabels([])
	#ax.get_xaxis().set_visible(False)
	#plt.xticks(rotation=45)
	#plt.xscale('log')
	plt.show()


#################################
########  A R  3 9 #############
#################################
print("depleted ar39...")
fileObject = open('valuesPhotonsPlot_Depleted39_SHAPE_299_300_Ntons_'+str(whichNtons)+'_pickle.txt','rb')
valuesPhotonsPlot39 = pickle.load(fileObject)
#valuesPhotonsPlot39 = loadtxt('valuesPhotonsPlot_39_SHAPE_299_300_Ntons_'+str(whichNtons)+'.txt') # np.shape = (299,300)
fileObject = open('photonsArr_flux_per_tSNburst_39_pickle.txt','rb')
photons39 = pickle.load(fileObject)
#photons39 = loadtxt('photonsArr_flux_per_tSNburst_39.txt') #np.shape = 300

photons39_1x1 = np.arange(min(photons39),max(photons39),1)
print('photons39_1x1: ',np.shape(photons39_1x1))
f_39 = scipy.interpolate.interp2d(time,photons39,valuesPhotonsPlot39.T)
print("done interpolating valuesPhotonsPlot39")
valuesPhotonsPlot39_1x1 = f_39(time,photons39_1x1) # by '1x1 I just mean 1-photon binning, but log time as usual
valuesPhotonsPlot39_1x1_t30 = f_39(time_30,photons39_1x1)

if boolPlotPreSmear:
	plt.title("Ar39 Photons vs. Time (Pre-Smear)")
	X,Y = np.meshgrid(time_30,photons39_1x1)
	plt.pcolormesh(X,Y, valuesPhotonsPlot39_1x1_t30)
	plt.colorbar(label="events/Nph/sec")
	plt.xlabel("time (s)")
	plt.ylabel("photons")
	#plt.xscale('log')
	ax = plt.gca()
	ax.set_xticks(time_30)
	ax.set_xticklabels([])
	ax.grid()
	plt.show()


###################################
######  Load Transfer Matrix  #####
###################################
whichNumFiles = 1337 #used to access the transfer matrix with numFiles incorporated (numFiles*10sources/file = sources used)

print("Loading smearnew_39_Files_"+str(whichNumFiles)+".txt")
thenLoad = TIME.time()
#smearfull = loadtxt("smearnew_39_Files_"+str(whichNumFiles)+".txt") # this matrix is 1x1 binning, generated from Use_transfer_matrix...py 
fileObject = open('smearnew_39_Files_'+str(whichNumFiles)+'_pickle.txt','rb')#open file for reading
smearfull = pickle.load(fileObject)#load the matrix from file into 'smear' var

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

#\\\\ Plot the Transfer Matrix (that's Ar39 range) ///
print("trying to plot transfer matrix...")
if boolPlotSmear39:
	#plt.title("Transfer Matrix: Nph_det vs. Nph_prod")
	#X,Y = np.meshgrid(photons39_1x1,photons39_1x1)
	#plt.pcolormesh(X,Y,smear39,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
	#plt.colorbar(label="probability")
	#plt.xlabel("Nph_produced")
	#plt.ylabel("Nph_detected")
	#plt.xlim(0,max(photons_1x1))
	#plt.ylim(0,max(photons_1x1))
	#plt.show()

	f_smear = scipy.interpolate.interp2d(photons39_1x1,photons39_1x1,smear39)
	xnew_LOG = np.logspace(log10(1),log10(max(photons39_1x1)),num=1000,base=10)
	xnew_LOG  = np.array( [0] + xnew_LOG.tolist()) #adding 0 photons to start of list
	ynew_LOG  = xnew_LOG
	smear39_LOG = f_smear(xnew_LOG,ynew_LOG)
	plt.title("Transfer Matrix: Nph_det vs Nph_prod")
	X,Y = np.meshgrid(xnew_LOG,ynew_LOG)
	plt.pcolormesh(X,Y,smear39_LOG.T,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-10))
	plt.colorbar(label="probability")
	plt.xlabel("Nph_produced")
	plt.ylabel("Nph_detected")
	plt.xlim(0,max(xnew_LOG))
	plt.ylim(0,max(ynew_LOG))
	plt.show()


print("smear39.T: ",np.shape(smear39), "  valuesPhotonsPlot39_1x1: ",np.shape(valuesPhotonsPlot39_1x1))
valuesPhotonsPlot39_1x1_smeared = np.matmul(smear39.T,valuesPhotonsPlot39_1x1)
print("valuesPhotonsPlot39_1x1_smeared: ",np.shape(valuesPhotonsPlot39_1x1_smeared))

#\\\\\ Suppress negative #s and #s < 10^-10 /////////
for i in range(len(valuesPhotonsPlot39_1x1_smeared)):
	for j in range(len(valuesPhotonsPlot39_1x1_smeared[0])):
		if valuesPhotonsPlot39_1x1_smeared[i][j] < 0 or valuesPhotonsPlot39_1x1_smeared[i][j] < (10**-10):
			valuesPhotonsPlot39_1x1_smeared[i][j] = 0

f_39_smeared = scipy.interpolate.interp2d(time,photons39_1x1,valuesPhotonsPlot39_1x1_smeared)
valuesPhotonsPlot39_1x1_t30_smeared = f_39_smeared(time_30,photons39_1x1) 

if boolPlotPostSmear:
        plt.title("Ar39 Photons vs. Time (Smeared)")
        X,Y = np.meshgrid(time_30,photons39_1x1)
        plt.pcolormesh(X,Y, valuesPhotonsPlot39_1x1_t30_smeared)
        plt.colorbar(label="events/Nph/sec")
        plt.xlabel("time (s)")
        plt.ylabel("photons")
        #plt.xscale('log')
        #plt.ylim(0,1000)
        ax = plt.gca()
        ax.set_xticks(time_30)
        ax.set_xticklabels([])
        ax.grid()
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


f_CEvNS_smeared = scipy.interpolate.interp2d(time,photons_1x1,valuesPhotonsPlot_1x1_smeared)
valuesPhotonsPlot_1x1_t30_smeared = f_CEvNS_smeared(time_30,photons_1x1)

if boolPlotPostSmear:
        plt.title("CEvNS Photons vs. Time (Smeared)")
        X,Y = np.meshgrid(time_30,photons_1x1)
        plt.pcolormesh(X,Y, valuesPhotonsPlot_1x1_t30_smeared)
        plt.colorbar(label="events/Nph/sec")
        plt.xlabel("time (s)")
        plt.ylabel("photons")
        #plt.xscale('log')
        #plt.ylim(0,1000)
        ax = plt.gca()
        ax.set_xticks(time_30)
        ax.set_xticklabels([])
        ax.grid()
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


valuesPhotonsPlotCombo_1x1_smeared = np.copy(valuesPhotonsPlot39_1x1_smeared + valuesPhotonsPlot_1x1_smeared_padded)
print('valuesPhotonsPlotCombo_1x1_smeared: ',np.shape(valuesPhotonsPlotCombo_1x1_smeared))

#\\\\\ Changing from rate to count (so z axis reports bin-specific values) //////
valuesPhotonsPlotCombo_1x1_smeared_counts = np.copy(valuesPhotonsPlotCombo_1x1_smeared)
valuesPhotonsPlot_1x1_smeared_padded_counts = np.copy(valuesPhotonsPlot_1x1_smeared_padded)
valuesPhotonsPlot39_1x1_smeared_counts = np.copy(valuesPhotonsPlot39_1x1_smeared)


print("valuesPhotonsPlotCombo_1x1_smeared_counts: ",np.shape(valuesPhotonsPlotCombo_1x1_smeared_counts))
print("valuesPhotonsPlot_1x1_smeared_padded_counts: ",np.shape(valuesPhotonsPlot_1x1_smeared_padded_counts))
print("valuesPhotonsPlot39_1x1_smeared_counts: ",np.shape(valuesPhotonsPlot39_1x1_smeared_counts))
print("time: ",np.shape(time))
print("Changing to count. Need to ensure time & values_counts have same len...\ntime: ",np.shape(time),"  valuesPhotonsPlotCombo_1x1_smeared_counts: ",np.shape(valuesPhotonsPlotCombo_1x1_smeared_counts))

for i in range(len(valuesPhotonsPlotCombo_1x1_smeared_counts)): #bc values has dimensions (nph, time) & I want time slices
	#since Nph bins = 1x1, mult. by size of Nph bin = *1 = no need to mult.
	#multiply each bin in horizontal time slice by size of time bin
	for j in range(len(valuesPhotonsPlotCombo_1x1_smeared_counts[0])):
		valuesPhotonsPlotCombo_1x1_smeared_counts[i][j] *= time[j]			
		valuesPhotonsPlot_1x1_smeared_padded_counts[i][j] *= time[j]
		valuesPhotonsPlot39_1x1_smeared_counts[i][j] *= time[j]

if np.array_equal(valuesPhotonsPlot_1x1_smeared_padded_counts,valuesPhotonsPlot_1x1_smeared_padded):
	print("ERROR: valuesPhotonsPlot_1x1_smeared_padded_counts = valuesPhotonsPlot_1x1_smeared_padded !!!!!!")
if np.array_equal(valuesPhotonsPlot39_1x1_smeared,valuesPhotonsPlot39_1x1_smeared_counts):
	print("ERROR: valuesPhotonsPlot39_1x1_smeared_counts = valuesPhotonsPlot39_1x1_smeared !!!!!!!")
if np.array_equal(valuesPhotonsPlotCombo_1x1_smeared_counts,valuesPhotonsPlotCombo_1x1_smeared):
	print("ERROR: valuesPhotonsPlotCombo_1x1_smeared_counts = valuesPhotonsPlotCombo_1x1_smeared !!!!!!")

if boolPlotPostSmearCombo:
        plt.title("CEvNS+Ar39 (Combo)  Photons vs. Time (Smeared)")
        X,Y = np.meshgrid(time,photons39_1x1)
        plt.pcolormesh(X,Y, valuesPhotonsPlotCombo_1x1_smeared)
        plt.colorbar(label="events/Nph/sec")
        plt.xlabel("time (s)")
        plt.ylabel("photons")
        plt.xscale('log')
        plt.show()

############# Calculating ratio matrix ########################
print('ratio plot...')

ratioPlot_rate = valuesPhotonsPlot_1x1_smeared_padded / valuesPhotonsPlot39_1x1_smeared
ratioPlot_counts = np.copy(ratioPlot_rate)

for i in range(len(ratioPlot_counts)):
	for j in range(len(ratioPlot_counts[0])):
		ratioPlot_counts[i][j] *= time[j]

for i in range(len(ratioPlot_rate)):
	for j in range(len(ratioPlot_rate[0])):
		if np.isnan(ratioPlot_rate[i][j]): ratioPlot_rate[i][j] = 0

f_ratio_rate = scipy.interpolate.interp2d(time,photons39_1x1,ratioPlot_rate,kind='cubic') #interp the rate, not the counts
ratioPlot_rate_t30 = f_ratio_rate(time_30,photons39_1x1)
ratioPlot_counts_t30 = f_ratio_rate(time_30,photons39_1x1) * time_30_binsize *1 #times bin size = ~0.3sec * 1photon

### THIS IS WRONG!! multiplying by time_30[j] instead of by time bin (which is constant)
#for i in range(len(ratioPlot_counts_t30)):
#	for j in range(len(ratioPlot_counts_t30[0])):
#		ratioPlot_counts_t30[i][j] *= time_30[j]
print("RatioPlot_counts_t30\n\n:",ratioPlot_counts_t30)
print('~~~~~~ ratioPlot_counts_t30: ',np.shape(ratioPlot_counts_t30))
#plt.plot(ratioPlot_counts_t30[10])
#plt.show()

if boolPlotRatio:
        if boolRate:
                plt.clf()
                plt.title("Ratio (RATE) in Photons vs. Time \n(1x1) (Smeared) \n($\\frac{s}{b} = \\frac{CEvNS}{Ar39}$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, ratioPlot_rate, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5)))
                plt.colorbar(label="$\\frac{s}{b}$")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.title("Ratio (RATE) in Photons vs. Time \n(1x1) (Smeared) \n($\\frac{s}{b} = \\frac{CEvNS}{Ar39}$)")
                X,Y = np.meshgrid(time_30,photons39_1x1)
                plt.pcolormesh(X,Y, ratioPlot_rate_t30, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5)))
                plt.colorbar(label="$\\frac{s}{b}$")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                #plt.xscale('log')
                ax = plt.gca()
                ax.set_xticks(time_30)
                ax.set_xticklabels([])
                plt.show()

                plt.title("Ratio (RATE) in Photons vs. Time \n(1x1) (Smeared) \n($\\frac{s}{b} = \\frac{CEvNS}{Ar39}$)")
                X,Y = np.meshgrid(time_30,photons39_1x1)
                plt.pcolormesh(X,Y, ratioPlot_rate_t30, cmap='bwr',vmin=0)
                plt.colorbar(label="$\\frac{s}{b}$ (linear z)")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                #plt.xscale('log')
                ax = plt.gca()
                ax.set_xticks(time_30)
                ax.set_xticklabels([])
                plt.show()

        if boolCounts:
                '''
                plt.clf()
                plt.title("Ratio (COUNTS) in Photons vs. Time \n (Smeared) \n($\\frac{s}{b} = \\frac{CEvNS}{Ar39}$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, ratioPlot_counts, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5)))
                plt.colorbar(label="$\\frac{s}{b}$")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                #plt.xscale('log')
                ax = plt.gca()
                ax.set_xticks(time)
                ax.grid()
                ax.set_xticklabels([])
                plt.show()
                '''
                plt.clf()
                plt.title("Ratio (COUNTS) in Photons vs. Time \n (Smeared) \n($\\frac{s}{b} = \\frac{CEvNS}{Ar39}$)")
                X,Y = np.meshgrid(time_30,photons39_1x1)
                plt.pcolormesh(X,Y, ratioPlot_counts_t30, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-15)))
                plt.colorbar(label="$\\frac{s}{b}$")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                #plt.xscale('log')
                ax = plt.gca()
                ax.set_xticks(time_30)
                ax.grid()
                ax.set_xticklabels([])
                plt.show()
                plt.clf()

                plt.clf()
                plt.title("Ratio (COUNTS) in Photons vs. Time \n (Smeared) \n($\\frac{s}{b} = \\frac{CEvNS}{Ar39}$)")
                X,Y = np.meshgrid(time_30,photons39_1x1)
                plt.pcolormesh(X,Y, ratioPlot_counts_t30, cmap='bwr')
                plt.colorbar(label="$\\frac{s}{b}$ (linear z)")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                #plt.xscale('log')
                ax = plt.gca()
                ax.set_xticks(time_30)
                ax.grid()
                ax.set_xticklabels([])
                plt.show()
                plt.clf()

############# Calculating residual matrix #####################
'''
print("residual plot...")

residualPlot_rate = valuesPhotonsPlot_1x1_smeared_padded - valuesPhotonsPlot39_1x1_smeared

#\\\\\ Changing from rate to count (so z axis reports bin-specific values) //////
residualPlot_counts = np.copy(residualPlot_rate)
print("Changing to count. Need to ensure time & values_counts have same len...\ntime: ",np.shape(time),"  residualPlot_counts: ",np.shape(residualPlot_counts))
print("residualPlot_counts: ",np.shape(residualPlot_counts))

for i in range(len(residualPlot_counts)):
        for j in range(len(residualPlot_counts[0])):
	#since Nph bins = 1x1, mult. by size of Nph bin = *1 = no need to mult.
        #multiply each bin in horizontal time slice by size of time bin
        	residualPlot_counts[i][j] *= time[j]   

if np.array_equal(residualPlot_rate,residualPlot_counts):print("ERROR: resPlot_rate = resPlot_counts!!!")


if boolPlotResidual:
        if boolRate:
                plt.clf()
                plt.title("Residual (RATE) in Photons vs. Time \n(1x1) (Smeared) \n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_rate, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5)))
                plt.colorbar(label="events/Nph/sec")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (RATE) in Photons vs. Time \n(1x1) (Smeared) (<0-suppressed)\n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_rate, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=0))
                plt.colorbar(label="events/Nph/sec")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (RATE) in Photons vs. Time \n(1x1) (Smeared) \n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_rate, cmap='bwr')
                plt.colorbar(label="events/Nph/sec")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (RATE) in Photons vs. Time \n(1x1) (Smeared) (<0 suppressed) \n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_rate, cmap='bwr',vmin=0)
                plt.colorbar(label="events/Nph/sec")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

        if boolCounts:
                plt.clf()
                plt.title("Residual (COUNTS) in Photons vs. Time \n(1x1) (Smeared) \n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_counts, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5)))
                plt.colorbar(label="events")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (COUNTS) in Photons vs. Time \n(1x1) (Smeared) (<0-suppressed) \n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_counts, cmap='bwr',norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=0))
                plt.colorbar(label="events")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (COUNTS) in Photons vs. Time \n(1x1) (Smeared) \n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_counts, cmap='bwr')
                plt.colorbar(label="events")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()

                plt.clf()
                plt.title("Residual (COUNTS) in Photons vs. Time \n(1x1) (Smeared) (<0 suppressed)\n($s - b = CEvNS - Ar39$)")
                X,Y = np.meshgrid(time,photons39_1x1)
                plt.pcolormesh(X,Y, residualPlot_counts, cmap='bwr',vmin=0)
                plt.colorbar(label="events")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                plt.xscale('log')
                plt.show()
'''

############### Calculating Sigma matrix  #####################
# sigma for each bin = Signal / sqrt(signal +background)
print("sigma plot...")
sigmaPlot_rate = np.copy(valuesPhotonsPlot_1x1_smeared_padded)

sigmaPlot_counts = np.copy(sigmaPlot_rate)
for i in range(len(sigmaPlot_counts)):
	for j in range(len(sigmaPlot_counts[0])):	#mult. by bin size in Nph (bin=1) and time (log)
		sigmaPlot_counts[i][j] *= time[j]

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
sigmaPlot_counts = calc_sigmaPlot(sigmaPlot_counts,valuesPhotonsPlot_1x1_smeared_padded_counts,valuesPhotonsPlot39_1x1_smeared_counts)

f_sigma_rate = scipy.interpolate.interp2d(time,photons39_1x1,sigmaPlot_rate)
sigmaPlot_counts_t30 = f_sigma_rate(time_30,photons39_1x1) * time_30_binsize *1 #times bin size = ~0.3sec * 1photon
### THIS IS WRONG... was multiplying by time_30[j] instead of time bin size (constant)
#for i in range(len(sigmaPlot_counts_t30)):
#	for j in range(len(sigmaPlot_counts_t30[0])):
#		sigmaPlot_counts_t30[i][j] *= time_30[j]

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
        if boolCounts:
                plt.clf()
                plt.title("Sigma ($\sigma$) (COUNTS) in Photons vs. Time  \n$\\frac{s}{\sqrt{s+b}}$")
                X,Y = np.meshgrid(time_30,photons39_1x1)
                plt.pcolormesh(X,Y, sigmaPlot_counts_t30)
                plt.colorbar(label="$ \\frac{s}{\sqrt{s+b}}$")
                plt.xlabel("time (s)")
                plt.ylabel("photons")
                #plt.xscale('log')
                ax = plt.gca()
                ax.set_xticks(time_30)
                ax.grid()
                ax.set_xticklabels([])
                plt.show()

#add sigmas of each bin in quadrature
def calc_deltaChiSquared(sigmaPlot):
	deltaChiSquared = 0
	for i in sigmaPlot.flatten():
		deltaChiSquared += i**2
	deltaChiSquared = sqrt(deltaChiSquared)
	print('deltaChiSquared: ',deltaChiSquared)

print("From rate sigma: ",calc_deltaChiSquared(sigmaPlot_rate))
print("From counts sigma: ",calc_deltaChiSquared(sigmaPlot_counts))
print("From counts sigma t30: ",calc_deltaChiSquared(sigmaPlot_counts_t30))
nowAll = TIME.time()
print("DONE... Duration: ",round(nowAll-thenAll,2),' sec')
	
