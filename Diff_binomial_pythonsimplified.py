
# coding: utf-8

# In[5]:


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
from scipy.stats import binom
import scipy
import random


lowerIndex=0
upperIndex=299

Ntons = 10000 #10kT normalization
LYar = 24000 #lightyield 24,000 ph/MeV
LYsc = 10000 #lightyield 10,000 ph/MeV

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~ FOR CEvNS interactions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
Eee,dNdEee,EeeTot,photonTot,Nevents,time,dt = [], [], [], [], [], [], []       

for index in range(lowerIndex, upperIndex):
    
    datat = loadtxt("garching_pinched_info_key.dat")
    time.append(datat[index-lowerIndex,1])
    dt.append(datat[index-lowerIndex,2])
    
    data = loadtxt("./out/supernova_diff_rates-Ar40-helm-photons2-{}.out".format(index),float)
        #Eee (MeV)  &   dNdEee1  (& dNdEee2...) &  totquenchedevents (same as dNdEee unless in scint w/ many isotopes), all with bin step of 0.0001 MeV
      
    Eee.append(data[:, 0]) # in MeV
    dNdEee.append(np.array(data[:, -1])*Ntons) # per MeV/sec  #NORMALIZE TO 10kT

    eeetot=0
    step = 0.0001 #in MeV
    for i in range(len(Eee[index])):
        eeetot += Eee[index-lowerIndex][i]*dNdEee[index-lowerIndex][i]*step #totquenchedevents*enee*eneestep
    EeeTot.append(eeetot*dt[index-lowerIndex])
    photonTot.append(eeetot*dt[index-lowerIndex]*LYar)#photons measured from total "Eee" deposition in slice "dt"

print("TOTAL Ar40 CEvNS ENERGY OVER BURST: ",sum(EeeTot), " MeV")
print("TOTAL Ar40 CEvNS PHOTONS OVER BURST: ",sum(photonTot), " photons")


#Convert the lists into 1D numpy arrays
Eee = np.array(Eee)
photons = np.array(Eee*LYar)
dNdEee = np.array(dNdEee)
time = np.array(time)
dt = np.array(dt)

for listentry in dNdEee:
    for element in listentry:
        Nevents.append(element) #dNdEee in long 1D list = # of events/sec/MeV at that Eee level
Nevents = np.array(Nevents) #[t0E0, t0E1,...t0En, ..., tnE0, ... tnEn]

valuesEee = np.array(Nevents) # events/MeV/sec
valuesPhotons = valuesEee/LYar # events/Nph/sec

valuesPhotonsPlot = [] # 2D event array for CEvNS in Ar40
for i in range(len(time)):
    start=len(Eee[0])*i
    stop=len(Eee[0])*i + len(Eee[0])
    valuesPhotonsPlot.append(valuesPhotons[start:stop])
valuesPhotonsPlot = np.array(valuesPhotonsPlot)

sumt=0
for i in range(len(dt)):
    sumt+=valuesPhotonsPlot[i]*dt[i]*photons[i]*(photons[i][1]-photons[i][0])
print("SUM  = ",sum(sumt))


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~ FOR LAr Radiologicals (Ar-39 beta decay) ~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
data = loadtxt("ar39pointslow.txt") #  energy(keV)  &   counts/keV/sec(per kg)
ar39energydata = np.array(data[:,0])/1000 #MeV
ar39eventsdata = np.array(data[:,1])*1000  #counts/MeV/sec (per kg)
f39 = interp1d(ar39energydata,ar39eventsdata,'cubic') #given MeV energy, returns Ar-39 rate counts/MeV/sec/kg

nenbins39 = 300 #number of energy bins
energy39 = np.array(linspace(min(ar39energydata),max(ar39energydata),nenbins39))
nevents39 = np.array(f39(energy39))

energystep39 = energy39[1]-energy39[0]

def get39integral(nevents39Arr,energy39Arr):
    ensum=0
    energystep39 = energy39Arr[1]-energy39Arr[0]
    for i in range(len(nevents39Arr)):
        ensum += energystep39*nevents39Arr[i]*energy39Arr[i]
    return ensum

nevents39 /= get39integral(nevents39,energy39)  #normalize 
nevents39 *= 1000 #normalize integral to 1000 events/sec (per ton = 1000kg) * Ntons
nevents39 *= Ntons #normalize to 10kT #counts/MeV/sec (per 10kT)
photons39 = energy39*LYar
valuesPhotons39 = nevents39/LYar #counts/Nph/sec (per 10kT) (and normalized)

valuesPhotonsPlot39 = [] # 2D event array for Ar-39 decay in Ar40
for i in range(len(time)):
    valuesPhotonsPlot39.append(valuesPhotons39)
valuesPhotonsPlot39 = np.array(valuesPhotonsPlot39)

photonTot39 = [] #will be in (logarithmic) dt slices 
photonTot39test = []
for i in range(len(time)):
    phtot=0
    for j in range(len(photons39)):
        phtot += photons39[j]*valuesPhotonsPlot39[i][j]*(energystep39*LYar) #Nph * events/Nph/sec * photonstep
    photonTot39test.append(phtot)
    photonTot39.append(phtot*dt[i])#photons measured from total "Eee" deposition in slice "dt"
print("Shape photonTot39: ",np.shape(photonTot39))
print("Sum photonTot39: ",sum(photonTot39))



##########################################
# FOR GLEB COMPARISONS #
##########################################

print("FOR GLEB COMPARISONS\n")
from scipy.interpolate import interp1d
wavelengthdistr = loadtxt("larscintillationwavelengthdistr.txt")
rlengthvswlengh = loadtxt("rayleighlengthvswavelength.txt")
rlengthdistr = loadtxt("rayleighlengthdistr.txt")
wlength_distr, wlengthcount_distr = wavelengthdistr[:,0], wavelengthdistr[:,1]
wlength, rlength = rlengthvswlengh[:,0],rlengthvswlengh[:,1]
rlength_distr, rlengthcount_distr = rlengthdistr[:,0],rlengthdistr[:,1]

f_wlength = interp1d(wlength_distr,wlengthcount_distr)

def n_refrac(lam):
    '''for a given wavelength (nm), returns the index of refraction in Liquid Argon in scintillation region (arXiv:1502.04213)'''
    a0,aUV,aIR = 1.26,0.23,0.0023
    lamUV,lamIR = 106.6,908.3
    nsquared = a0 + (aUV*lam**2)/(lam**2 - lamUV**2) + (aIR*lam**2)/(lam**2-lamIR**2)
    return sqrt(nsquared)

def lar_cauchy(x):
    '''given scintillation wavelength (nm), returns the LAr wavelength emission spectrum as a Cauchy distribution'''
    sigma, peak = 3.2,128 #no units, nanometers
    hwhm = (sigma*2.355)/2 #FWHM/2
    return 1/(pi*hwhm*(1+((x-peak)/hwhm)**2))

def lar_gaussian(x):
    '''given scintillation wavelength (nm), returns LAr wavelength emission spectrum (values 0-1) as Gaussian'''
    sigma, mu = 3.2,128 #no units, nanometers
    coeff=1/sqrt(2*pi*sigma**2)
    return coeff * exp(-(x-mu)**2/ (2*sigma**2))

def rlength_cauchy(x):
    '''given Rayleigh length (cm), returns Rayleigh length spectrum (values 0-1) as Cauchy distr'''
    sigma, peak = 20.23, 60 #63.74
    hwhm = (sigma*2.355)/2 #FWHM/2
    return 1/(pi*hwhm*(1+((x-peak)/hwhm)**2))

def rlength_gaussian(x):
    '''given Rayleigh length (cm), returns Rayleigh length spectrum (values 0-1) as Gaussian'''
    sigma, mu = 20.23, 61 #63.74
    coeff=1/sqrt(2*pi*sigma**2)
    return coeff * exp(-(x-mu)**2/ (2*sigma**2))

n_refrac = vectorize(n_refrac)
lar_cauchy = vectorize(lar_cauchy)
lar_gaussian = vectorize(lar_gaussian)
rlength_cauchy = vectorize(rlength_cauchy)
rlength_gaussian = vectorize(rlength_gaussian)

print("~~~~~~~~~~~~~~~~~~~ (done first block)\n")


def D_diffconst(rayleighlength,absorptionlength):
    '''given rlength and ablength in cm, returns diffusivity constant'''
    ma, ms, g = 1/absorptionlength, 1/rayleighlength, 0.025
    return 1/(3*(ma+(1-g)*ms))

def absorpcoeff(absorptionlength,t):
    ma = 1/absorptionlength
    return exp(-ma*c*t)

def coeff(t): #1D version
    return 1/(4*pi*DC*t)**(1/2)

def Sx(x,x0,w,t):
    '''Gaussian sum for 1D (no coeff)'''
    sumsx = 0
    for n in range(-N,N+1):
        sumsx += exp(-(x-x0 +4*n*w)**2/(4*DC*t)) - exp(-(x+x0+(4*n-2)*w)**2/(4*DC*t))
    return sumsx
def dxSx(x,x0,w,t):
    '''evaluating dSx/dx at const boundary x (no coeff)'''
    sumsx = 0
    for n in range(-N,N+1): #d/dx g=exp(-s(x+w)^2) = -2s(x+w)g
        sumsx += (-2*(x-x0 +4*n*w)/(4*DC*t))*exp(-(x-x0 +4*n*w)**2/(4*DC*t)) #term 1
        sumsx += -(-2*(x+x0+(4*n-2)*w)/(4*DC*t))*exp(-(x+x0+(4*n-2)*w)**2/(4*DC*t)) #term 2
    return sumsx
def integSx(xa,xb,x0,w,t):
    '''integral of Sx over xa to xb (no coeff)'''
    sumsx = 0
    for n in range(-N,N+1):
        termwithxb = (1/2)*sqrt(pi*4*DC*t)*(erf((xb-x0+4*n*w)/sqrt(4*DC*t)) - erf((xb+x0+(4*n-2)*w)/sqrt(4*DC*t)))
        termwithxa = (1/2)*sqrt(pi*4*DC*t)*(erf((xa-x0+4*n*w)/sqrt(4*DC*t)) - erf((xa+x0+(4*n-2)*w)/sqrt(4*DC*t)))
        sumsx += termwithxb - termwithxa
    return sumsx
def pdensity(x,y,z,x0,y0,z0,w,l,h,t):
    return coeff(t)**3 * Sx(x,x0,w,t) * Sx(y,y0,l,t) * Sx(z,z0,h,t)

def pdensity_integ(x0,y0,z0):
    if selectSide=='x':
        wall = -xhold*sideSign
        pdensityintegrated_t = sideSign*dt*DC*absorpcoeff(absorptionlength,tarr)*(coeff(tarr)**3 * integSx(za,zb,z0,h,tarr) * integSx(ya,yb,y0,l,tarr) * dxSx(wall,x0,w,tarr)) 
    if selectSide=='y':
        wall = -yhold*sideSign
        pdensityintegrated_t = sideSign*dt*DC*absorpcoeff(absorptionlength,tarr)*(coeff(tarr)**3 * integSx(za,zb,z0,h,tarr) * integSx(xa,xb,x0,w,tarr) * dxSx(wall,y0,l,tarr)) 
    if selectSide=='z':
        wall = -zhold*sideSign
        pdensityintegrated_t = sideSign*dt*DC*absorpcoeff(absorptionlength,tarr)*(coeff(tarr)**3 * integSx(xa,xb,x0,w,tarr) * integSx(ya,yb,y0,l,tarr) * dxSx(wall,z0,h,tarr)) 
    return pdensityintegrated_t

#def binomcoeff(n,k):
#    '''returns the Ck_n binomial coefficient for ways of detecting k photons given n created photons'''
#    return factorial(n)//(factorial(k)*factorial(n-k)) #implementing integer division "//"

def binomdistr(n,k,x0,y0,z0,PDENdict):
    '''using scipy.stats.binom.pmf... given n=createdPhotons and k=detectedPhotons (k<=n) and source location, returns binomial distribution probability of k'''
    #prob_xyz = sum(pdensity_integ(x0,y0,z0)) #sum up all the tarr bins
    prob_xyz = sum(PDENdict[str(x0)+","+str(y0)+","+str(z0)])
    return binom.pmf(k,n,prob_xyz) #binomcoeff(n,k)*(prob_xyz**k)*(1-prob_xyz)**(n-k)

def integBinomDistr(n,k,spatialbins_x,spatialbins_y,spatialbins_z,PDENdict):
    '''returns average probability p(n,k) of getting k photons from n created photons'''
    '''integrates over all source locations in x,y,z and divides by volume'''
    dx0,dy0,dz0 = (xb-xa)/spatialbins_x,(yb-ya)/spatialbins_y,(zb-za)/spatialbins_z
    x0arr = linspace(xa+dx0/2,xb+dx0/2 -dx0,spatialbins_x) #place source in center of voxel
    y0arr = linspace(ya+dy0/2,yb+dy0/2 -dy0,spatialbins_y)
    z0arr = linspace(za+dz0/2,zb+dz0/2 -dz0,spatialbins_z)
    volume = (xb-xa)*(yb-ya)*(zb-za)
    sumx,sumy,sumz=0,0,0
    total=0
    for x0 in x0arr:
        for y0 in y0arr:
            for z0 in z0arr:
                total+=binomdistr(n,k,x0,y0,z0,PDENdict)*(dx0*dy0*dz0)
    return total/volume
    
def createPDENarr(xa,xb,ya,yb,za,zb,spatialbins_x,spatialbins_y,spatialbins_z):
    print("\nSTARTING createPDENarr : \n")
    then=TIME.time()
    dx0,dy0,dz0 = (xb-xa)/spatialbins_x,(yb-ya)/spatialbins_y,(zb-za)/spatialbins_z
    x0arr = linspace(xa+dx0/2,xb+dx0/2 -dx0,spatialbins_x) #place source in center of voxel
    y0arr = linspace(ya+dy0/2,yb+dy0/2 -dy0,spatialbins_y)
    z0arr = linspace(za+dz0/2,zb+dz0/2 -dz0,spatialbins_z)
    print("x0arr: ",x0arr)
    print("y0arr: ",y0arr)
    print("z0arr: ",z0arr)
    PDENdict = {}
    for x0 in x0arr:
        for y0 in y0arr:
            for z0 in z0arr:
                print(str(x0)+","+str(y0)+","+str(z0))
                PDENdict[str(x0)+","+str(y0)+","+str(z0)] = pdensity_integ(x0,y0,z0) 
    now=TIME.time()
    print("\nENDING createPDENarr... DURATION: ",now-then,"\n")
    print("Writing to file PDENdict.txt")
    pdendictfile = open("PDENdict2.txt",'w')
    for entry in PDENdict:
        pdendictfile.write('\n'+entry+'   ')
        for i in range(len(PDENdict[entry])):
            pdendictfile.write(str(PDENdict[entry][i])+"  ")
    print("Done writing")
    return PDENdict


absorpcoeff = vectorize(absorpcoeff)
pdensity=vectorize(pdensity)
integSx = vectorize(integSx)
dxSx = vectorize(dxSx)
Sx = vectorize(Sx)


rayleighlength, absorptionlength = 55, 20e2 #cm
D = D_diffconst(rayleighlength,absorptionlength) #cm
c = 30/n_refrac(128) #cm/ns
DC=D*c #cm^2/ns #diffusivity constant

# Setting locations and times
#D = 18.8 #cm #was 18.8
#c = 21.7 #cm/ns
#DC = D*c #cm^2/ns #diffusivity constant
lardensity = 1.346e-3 #kg/cm^3
t0=0 #assume instantaeous arrival to detectors: no integration over t0

print("D: ",D, "  c: ",c,"  DC: ",D*c)

xhold,yhold,zhold = 1.8e2,6e2,29e2 # 3.6m x 12 m x 58m
xa,xb,ya,yb,za,zb = -xhold,xhold,-yhold,yhold,-zhold,zhold
w,l,h = xhold,yhold,zhold
exten = 2.143*D
#exten = 1.714*D
w,l,h = w+exten,l+exten,h+exten
N = 10 #"infinite sum" for method of images, was 1000

dt=0.1 #dt was 0.5  # <<<<< CHANGE ME!
tarr = arange(0.01,500,dt) #was 500

#xarr = linspace(xa,xb,100)
#yarr = linspace(ya,yb,100)
#zarr = linspace(za,zb,100)

selectSide = 'x' # <<<<< CHANGE ME!
sideSign = -1 #1 or -1 #<<<<< WHICH +/- WALL
spatialbins_x,spatialbins_y,spatialbins_z=10,10,10
#x0,y0,z0=179,0,0 # <<<<< SOURCE LOCATION

print("\n~~~~~~~~~~(DONE initializing dimension variables)\n")





import random

#def bpmf(k,n,p): 
#    return scipy.special.comb(n, k)*(p**k)*((1-p)**(n-k))

def binomdistr_random(n,k,x0,y0,z0,PDENdict_random):
    '''using scipy.stats.binom.pmf... given n=createdPhotons and k=detectedPhotons (k<=n) and source location, returns binomial distribution probability of k'''
    #prob_xyz = sum(pdensity_integ(x0,y0,z0)) #sum up all the tarr bins
    

    
    prob_xyz = PDENdict_random[str(x0)+","+str(y0)+","+str(z0)] # was sum(PDENdict_random[])...
    
    # hopefully none of these trigger ever again
    #if prob_xyz>1:
    #    print("    ############### PROB > 1 ... p=",prob_xyz," ... ",x0,y0,z0)
    #    prob_xyz = 1 #trying to fix "nan" issue by fixing max probability
    #if np.isnan(prob_xyz):
    #    print("    ######## INSIDE binomdistr_random() --> prob_xyz = ",prob_xyz)
        
    b = binom.pmf(k,n,prob_xyz)

    #if np.isnan(b):
    #    print("    ######## INSIDE binomdistr_random() --> binom.pmf(k=",k,",n=",n,",p=",prob_xyz,") = ",b)
    #    b = bpmf(k,n,prob_xyz)
    #if np.isnan(b):
    #    print("    ((( ######## INSIDE binomdistr_random() --> homemade bpmf(k,n,prob_xyz) = ",b)
    
    return b

    #return binom.pmf(k,n,prob_xyz) #binomcoeff(n,k)*(prob_xyz**k)*(1-prob_xyz)**(n-k)

def integBinomDistr_random(n,k,sourcearr,PDENdict_random):
    '''returns average probability p(n,k) of getting k photons from n created photons'''
    '''integrates over all source locations in x,y,z and divides by volume'''
    volume = (xb-xa)*(yb-ya)*(zb-za)
    total=0
    for loc in sourcearr:
        total+=binomdistr_random(n,k,loc[0],loc[1],loc[2],PDENdict_random) #not multiplying by dx0dy0dz0...
        #if np.isnan(total):
        #    print("######## INSIDE integBinomDistr_random() --> last addition, binomdistr_random = ",binomdistr_random(n,k,loc[0],loc[1],loc[2],PDENdict_random))
    #if len(sourcearr)==0:
    #    print("######## INSIDE integBinomDistr_random() --> len(sourcearr) = 0")
    #if np.isnan(total/len(sourcearr)):
    #    print("######## INSIDE integBinomDistr_random() --> total/len(sourcearr) is NAN")
    #    print("     total: ",total)
    #    print("     len(sourcearr): ",len(sourcearr))

    return total/len(sourcearr) #not dividing by volume...

def createPDENarr_random(xa,xb,ya,yb,za,zb):
    '''given detector dimensions & number of random sources, returns dictionary of prob densities & the array of source locations'''
    print("\n@ CreatePDENarr_random()...")
    then=TIME.time()
    sourcearr = []
    numsources = 1 #random source locations
    
    for i in range(numsources):
        x0=random.randint(xa,xb)
        y0=random.randint(ya,yb)
        z0=random.randint(za,zb)        
        sourcearr.append([x0,y0,z0])
        print("  ",i+1,"out of ",numsources,"      \\\ ",x0,y0,z0," ///")
    
    PDENdict_random = {} #key = string of source coordinates, value = scalar probability (sum of all tarr bins in pdensity_integ() array)
    for loc in sourcearr:
        if (loc[0]>160): #if x is close to wall, do a linear extrapolation to obtain probability
            prob150 = sum(pdensity_integ(150,loc[1],loc[2])) #sum up all the tarr bins
            prob155 = sum(pdensity_integ(155,loc[1],loc[2]))
            extrapEqn = scipy.interpolate.InterpolatedUnivariateSpline([150,155],[prob150,prob155],k=1)
            prob_xyz = extrapEqn([loc[0]])[0]
            print("   ** Extrap prob for x=",loc[0]," :  ",prob_xyz)
        else:
            prob_xyz = sum(pdensity_integ(loc[0],loc[1],loc[2])) #sum up all the tarr bins
        
        PDENdict_random[str(loc[0])+","+str(loc[1])+","+str(loc[2])] = prob_xyz #was:  = pdensity_integ(loc[0],loc[1],loc[2]), an array

    now=TIME.time()
    print("   @ DURATION: ",now-then)
    return PDENdict_random, sourcearr


def createsmearedmatrix_random(narr,PDENdict_random,sourcearr):
    '''for a given array of 'n' Nph_produced, returns the normalized 2D array of smeared probabilities for Nph_detected'''
    print("@ Createsmearedmatrix_random()...")
    print("   nmax, nlen: ",max(narr),", ",len(narr))
    thenCreate=TIME.time()
    #probnkarr_full = []
    probnkarr_normalized_full = []
    
    noNAN = True
    
    #counter = 0
    ratio15,ratio25,ratio35,ratio45 = int(len(narr)/5),int(2*len(narr)/5),int(3*len(narr)/5),int(4*len(narr)/5)
    for n in narr:
        #if counter == ratio15: print("  -- 1/5 thru narr (Duration from t0: ",TIME.time() - thenCreate," sec)")
        #if counter == ratio25: print("  -- 2/5 thru narr (Duration from t0: ",TIME.time() - thenCreate," sec)")
        #if counter == ratio35: print("  -- 3/5 thru narr (Duration from t0: ",TIME.time() - thenCreate," sec)")
        #if counter == ratio45: print("  -- 4/5 thru narr (Duration from t0: ",TIME.time() - thenCreate," sec)")
        
        Nph_prod = n
        kstep = 10
        ktop = Nph_prod #Nph_prod + 1
        karr = arange(0,ktop,kstep)

        if (karr.all() <ktop) or (not karr): 
            karr=np.concatenate((karr,[ktop]))

        probnkarr = []
        for k in karr:
            probnkarr.append(integBinomDistr_random(Nph_prod,k,sourcearr,PDENdict_random))
            
        kprobinterp = []
        if len(karr)>1: #can't interpolate with one item
            kprobinterp = interp1d(karr,probnkarr)
            probnkarr = kprobinterp(arange(0,ktop,1)) #no gaps, step thru by "1"
        while (len(probnkarr) < max(narr)):
            probnkarr = np.concatenate((probnkarr,[0]))

        #probnkarr=np.array(probnkarr)
        probnkarr_normalized=np.array(probnkarr)/sum(probnkarr) #normalize
        #probnkarr_full.append(probnkarr)
        probnkarr_normalized_full.append(probnkarr_normalized)
        #counter +=1
        
    nowCreate=TIME.time()
    print("   @ DURATION: ",nowCreate-thenCreate)
    #return probnkarr_full, probnkarr_normalized_full
    return probnkarr_normalized_full

def write_smearmatrix(smearmatrix):
    theTime = TIME.time()
    outfile = open("smearmatrix"+str(len(smearmatrix))+"_"+str(len(smearmatrix[0]))+"_TIME_"+str(theTime)+".txt",'w') 
    for i in smearmatrix:
        for j in i:
            outfile.write(str(j)+"  ")
        outfile.write("\n\n")
    outfile.close()
    print("Wrote to : \'smearmatrix"+str(len(smearmatrix))+"_"+str(len(smearmatrix[0]))+"_TIME_"+str(theTime)+".txt\'")

def run_smearcodeXtimes(x):
    smearX = []
    timePerSource = [] 
    thenOVERALL = TIME.time()
    for i in range(x):
        print("\n======================================\n===== ",i+1,"  out of  ",x,"  =====")
        thenRUN = TIME.time()
        PDENdict_random, sourcearr = createPDENarr_random(xa,xb,ya,yb,za,zb)

        narrmax = 1500 # MAX NPH_PROD TO CREATE TRANSFER MATRIX FOR
        narrstep = 10
        narr = arange(0,narrmax,narrstep) #Nph_produced array #was 1 spacing
        smearedprobs_normalized = createsmearedmatrix_random(narr,PDENdict_random,sourcearr)
        smearedprobs_normalized = array(smearedprobs_normalized)


        dimx,dimy=np.shape(smearedprobs_normalized),np.shape(smearedprobs_normalized[0])
        smearedprobs_normalized_reshape = np.concatenate(smearedprobs_normalized)
        smearedprobs_normalized_reshape = smearedprobs_normalized_reshape.reshape(dimx[0],dimy[0])
        
        smearX.append(smearedprobs_normalized_reshape)
        nowRUN = TIME.time()
        timePerSource.append(nowRUN - thenRUN) #includes time to make PDENarr and to create smeared matrix
        '''     
        if x==1:
            plt.title("Smeared Nph_det vs. Nph_prod")
            X,Y = np.meshgrid(narr,arange(min(narr),max(narr),1))
            plt.pcolormesh(X,Y,smearedprobs_normalized_reshape.T,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
            plt.colorbar(label="probability")
            plt.xlabel("Nph_produced")
            plt.ylabel("Nph_detected")
            plt.show()
        '''
    smearX = np.array(smearX)
    smeartotal = sum(smearX) #add the individual sources into one transfer matrix
    for i in range(len(smeartotal)): #normalize that matrix by vertical column (i.e. per Nph_prod slice)
        smeartotal[i] = smeartotal[i] / sum(smeartotal[i])
     
    write_smearmatrix(smeartotal) #write the matrix
    ''' #turning off plotting on desktop

    plt.title("Smeared Nph_det vs. Nph_prod")
    X,Y = np.meshgrid(narr,arange(min(narr),max(narr),1))
    plt.pcolormesh(X,Y,smeartotal.T,norm=mpl.colors.SymLogNorm(linthresh=(1e-5),vmin=1e-5))
    plt.colorbar(label="probability")
    plt.xlabel("Nph_produced")
    plt.ylabel("Nph_detected")
    plt.show()
    
    plt.title("Distribution of Time to Run Per Source")
    plt.xlabel("Individual 1-Source Runs")
    plt.ylabel("Time (sec)")
    plt.plot(timePerSource,'.')
    plt.show()
    '''
    print("\nAverage time per 1-source run: ",np.mean(np.array(timePerSource))," sec\n")
    nowOVERALL = TIME.time()
    print("Total time for run_smearcodeXtimes(",x,"): ",nowOVERALL-thenOVERALL," sec\n")

then=TIME.time()
numOfWrites = 600
numOfSourcesPerWrite = 10
for i in range(numOfWrites): #TURN WRITING BACK ON
    print(">>>>> ",i+1," writes out of ",numOfWrites," <<<<<")
    run_smearcodeXtimes(numOfSourcesPerWrite)
#run_smearcodeXtimes(100)
now=TIME.time()
print("\n>>>DURATION: ",now-then, " sec <<<")
print("DONE")




