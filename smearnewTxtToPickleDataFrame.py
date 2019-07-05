import pandas as pd
import numpy as np
import pickle
import time as TIME

filetopickle = 'valuesPhotonsPlot_Depleted39_SHAPE_299_300_Ntons_2500.txt'
pickledfile = filetopickle[:-4]+'_pickle.txt'

print("Loading via loadtxt...")
thenloadtxt = TIME.time()
matrixtopickle = np.loadtxt(filetopickle)
nowloadtxt = TIME.time()
print("Duration of loadtxt: ",round(nowloadtxt-thenloadtxt,2),' sec')

file_Name = pickledfile
fileObject = open(file_Name,'wb') #open file for binary writing

print("Pickling...")
thenpickle = TIME.time()
pickle.dump(matrixtopickle,fileObject) #writes smeartotal to file
nowpickle = TIME.time()
fileObject.close()
print("Duration of pickle.dump: ",round(nowpickle-thenpickle,2),' sec')

fileObject = open(file_Name,'rb')#open file for reading
print("Loading via pickle.load...")
thenloadpickle = TIME.time()
pickledmatrix = pickle.load(fileObject)#load the matrix from file into 'smear' var
nowloadpickle = TIME.time()
print("Duration of pickle.load: ",round(nowloadpickle-thenloadpickle,2),' sec')




