import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
import functionsNN as fNN
import random as rd
import collections
import time
drive = '/Volumes/LaCie/Data/'
filename = 'LoughranMcDonald_MasterDictionary_2014.csv'
dict_raw = pd.read_csv(drive+filename, index_col=0)
dict_raw.index = dict_raw.index.str.lower()
dict_raw['Negative'][dict_raw['Negative']!=0] = 1
dict_raw['Positive'][dict_raw['Positive']!=0] = 1
dict_raw = dict_raw[(dict_raw['Negative']+dict_raw['Positive']).ne(0)]
dict_raw = dict_raw[(dict_raw['Average Proportion']).ne(0)]
weight = dict_raw['Word Proportion']/dict_raw['Average Proportion']

masterDF = dict_raw.iloc[:,6:8]
masterDF = (masterDF.T*weight).T
score = masterDF['Positive']-masterDF['Negative']
masterDict = score.to_dict()
fd.saveFile(masterDict, drive + 'Loughran_McDonald_dict.pckl')