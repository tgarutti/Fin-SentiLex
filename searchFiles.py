import numpy as np
import re
import pandas as pd
import functions10X as f10X
import functionsData as fd
import functionsNN as fNN
import functionsSVM as fSVM
import random as rd
import collections
import time
drive = "/Volumes/LaCie/Data/"
search = []
wc = fd.loadFile(drive+'length.pckl')
pos10K = 0
neg10K = 0
pos10Q = 0
neg10Q = 0
wcPos10K = 0
wcNeg10K = 0
wcPos10Q = 0
wcNeg10Q = 0
for year in range(2000,2015):
        print(year)
        f1 = drive+str(year)+"10X_final.pckl"
        dataset = fd.loadFile(f1)
        for item in dataset:
            wc_cik = wc[(wc[:,2] == item[1])]
            wc_i = wc_cik[(wc_cik[:,0] == item[0])]
            count = sum(wc_i[:,1].astype(int))
            if item[5]>=0:
                if '10-K' in item[2]:
                    pos10K+=1
                    wcPos10K+=count
                elif '10-Q' in item[2]:
                    pos10Q+=1
                    wcPos10Q+=count
            if item[5]<0:
                if '10-K' in item[2]:
                    neg10K+=1
                    wcNeg10K+=count
                elif '10-Q' in item[2]:
                    neg10Q+=1
                    wcNeg10Q+=count
train = [pos10K,neg10K,pos10Q,neg10Q,wcPos10K/pos10K,wcNeg10K/neg10K,wcPos10Q/pos10Q,wcNeg10Q/neg10Q]
                    
pos10K = 0
neg10K = 0
pos10Q = 0
neg10Q = 0
wcPos10K = 0
wcNeg10K = 0
wcPos10Q = 0
wcNeg10Q = 0
for year in range(2015,2019):
        print(year)
        f1 = drive+str(year)+"10X_final.pckl"
        dataset = fd.loadFile(f1)
        for item in dataset:
            wc_cik = wc[(wc[:,2] == item[1])]
            wc_i = wc_cik[(wc_cik[:,0] == item[0])]
            count = sum(wc_i[:,1].astype(int))
            if item[5]>=0:
                if '10-K' in item[2]:
                    pos10K+=1
                    wcPos10K+=count
                elif '10-Q' in item[2]:
                    pos10Q+=1
                    wcPos10Q+=count
            if item[5]<0:
                if '10-K' in item[2]:
                    neg10K+=1
                    wcNeg10K+=count
                elif '10-Q' in item[2]:
                    neg10Q+=1
                    wcNeg10Q+=count
test = [pos10K,neg10K,pos10Q,neg10Q,wcPos10K/pos10K,wcNeg10K/neg10K,wcPos10Q/pos10Q,wcNeg10Q/neg10Q]
                     