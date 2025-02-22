##############################################################################
##### Get list of 10X files based on CIK and path length #####################
##############################################################################
import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
 
# Location external drive
drive = "/Volumes/LaCie"
test_name = "10-X_C_2016-2018"
folder_name = "10-X"
list10K = []
list10Q = []

for folder in reversed(os.listdir(drive)):
    if folder.startswith(folder_name):
        for year in reversed(os.listdir(drive+"/"+folder)):
            if year[0].isdigit():
                print(year)
                year10K = []
                year10Q = []
                for quarter in reversed(os.listdir(drive+"/"+folder+"/"+year)):
                    if quarter.startswith("QTR"):
                        for filename in os.listdir(drive+"/"+folder+"/"+year+"/"+quarter):
                            # Open file and load text
                            file_path = drive+"/"+folder+"/"+year+"/"+quarter+"/"+filename
                            text = open(file_path,"r").read()
                            
                            # Read header: get company name, CIK and file type
                            header = f10X.readHeader(text).tolist()
                            date = year + " " + quarter[0] + quarter[-1]
                            if f10X.checkDates(text, 360):
                                if "10-K" == header[-1]:
                                    year10K.append(np.array([date, header[0], header[1], header[2], file_path]))
                                elif "10-Q" == header[-1]:
                                    year10Q.append(np.array([date, header[0], header[1], header[2], file_path]))
                list10K.append(year10K)
                list10Q.append(year10Q)

## Sort list/arrays of 10Ks and 10Qs by the CIK
array10K = np.vstack(list10K)
array10K = array10K[array10K[:,2].argsort()]
array10Q = np.vstack(list10Q)
array10Q = array10Q[array10Q[:,2].argsort()]
cik_10Q = np.unique(array10Q[:,2])
cik_10K = np.unique(array10K[:,2])

## Delete for loop variables 
del folder, year, quarter, year10K, year10Q, file_path, text, header, date, list10K, list10Q
array10Q_final = []
array10K_final = []
cik_final = []
yrs_min = 18
yrs_test = 3

for i in cik_10Q:
    temp10Q = array10Q[array10Q[:,2]==i,:]
    temp10K = array10K[array10K[:,2]==i,:]
    qrt, indQ = np.unique(temp10Q[:,0], return_index=True)
    yr, indK = np.unique(temp10K[:,0], return_index=True)
    if qrt.shape[0] >= yrs_min*3 and yr.shape[0] >= yrs_min:
        array10Q_final.append(temp10Q[indQ,:])
        array10K_final.append(temp10K[indK,:])
        cik_final.append(i)
        
del yrs_min, yrs_test, yr, test_name, temp10Q, temp10K, qrt, indQ, indK, i, cik_10K, cik_10Q
array10Q_final = np.vstack(array10Q_final)
array10K_final = np.vstack(array10K_final)
list10Q = array10Q_final.tolist()
list10K = array10K_final.tolist()
CIKs = np.unique(array10Q_final[:,2]).tolist()
fileNames10Q = array10Q_final[:,4].tolist()
fileNames10K = array10K_final[:,4].tolist()
fd.saveFile(list10Q, "10Qs.pckl")
fd.saveFile(list10K, "10Ks.pckl")
fd.saveFile(CIKs, "CIKs.pckl")
fd.saveFile(fileNames10Q, "fileNames10Q.pckl")
fd.saveFile(fileNames10K, "fileNames10K.pckl")
del array10Q_final, array10K_final, drive, filename, folder_name, array10K, array10Q, cik_final