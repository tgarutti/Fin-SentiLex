##############################################################################
##### Get list of 10X files based on CIK and path length #####################
##############################################################################
import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
import math
from collections import defaultdict
loc = "/Volumes/LaCie/Data/"

def getList10X(folder_name, year):
    CIKs = []
    list10K = []
    list10Q = []
    for quarter in os.listdir(folder_name):
        if quarter.startswith("QTR"):
            for filename in os.listdir(folder_name+"/"+quarter):
                # Open file and load text
                file_path = folder_name+"/"+quarter+"/"+filename
                text = open(file_path,"r").read()
                
                # Read header: get company name, CIK and file type
                header = f10X.readHeader(text).tolist()
                date = year + " " + quarter[0] + quarter[-1]
                if f10X.checkDates(text, 360):
                    if len(f10X.checkItems(text)) > 10:
                        CIKs.append(header[1])
                        if "10-K" == header[-1]:
                            list10K.append([date, header[1], file_path, f10X.itemize10X(file_path)])
                        elif "10-Q" == header[-1]:
                             list10Q.append([date, header[1], file_path, f10X.itemize10X(file_path)])
    CIKs = np.unique(np.array(CIKs)).tolist()
    list10X = list10K + list10Q
    return list10X, CIKs

def dataToLists():
    drive = "/Volumes/LaCie"
    CIKs = []
    for folder in reversed(os.listdir(drive)):
        if folder.startswith("10-X"):
            for year in reversed(os.listdir(drive+"/"+folder)):
                if year[0].isdigit():
                    print(year)
                    list10X, CIKtemp = getList10X(drive+"/"+folder+"/"+year, year)
                    f1 = drive+"/Data/"+year+"10X.pckl"
                    fd.saveFile(list10X, f1)
                    CIKs = CIKs + CIKtemp
                    del list10X
    cik_set = set(CIKs)
    CIKs = list(cik_set)
    f2 = drive+"/Data/"+"CIKs.pckl"
    fd.saveFile(CIKs, f2)
                    
def readPrices(fn1, fn2, fn3, CIKs):
    p1 = pd.read_csv(fn1, dtype={"datadate": int, "prccm": float, "cik": int})
    p1 = p1[['datadate','prccm','cik']]
    p2 = pd.read_csv(fn2, dtype={"datadate": int, "prccm": float, "cik": int})
    p2 = p2[['datadate','prccm','cik']]
    p3 = pd.read_csv(fn3, dtype={"datadate": int, "prccd": float, "cik": int})
    p3 = p3[['datadate','prccd','cik']]
    p3.columns = ['datadate','prccm','cik']
    prices = pd.concat([p1,p2,p3])
    del p1,p2,p3 
    prices_final = []
    c = 0
    progress = 0.1
    for cik in CIKs:
        c+=1 
        if c/len(CIKs)>=progress:
            per = int(progress*100)
            print(str(per) + "%")
            progress+=0.1
        p_cik = prices.loc[prices['cik'] == int(cik)]
        for year in range(2000,2019):
            low = (year-1)*10000+1001
            high = year*10000+931
            p = p_cik.loc[(p_cik['datadate'] >= low) & (p_cik['datadate'] <= high)]
            p = p[p['prccm'].notna()]
            q1 = p.loc[(p['datadate']//100%100>=10) & (p['datadate']//100%100<=12)]
            q2 = p.loc[(p['datadate']//100%100>=1) & (p['datadate']//100%100<=3)]
            q3 = p.loc[(p['datadate']//100%100>=4) & (p['datadate']//100%100<=6)]
            q4 = p.loc[(p['datadate']//100%100>=7) & (p['datadate']//100%100<=9)]
            year_str = str(year)            
            qrt = [q1,q2,q3,q4]
            i = 0
            for q in qrt:
                i+=1
                a = q.empty
                if q.empty == False:
                    q = q.sort_values('datadate').iloc[-1]
                    date = year_str+" Q"+str(i)
                    d={'date':date,'price':q['prccm'],'cik':cik}
                    prices_final.append(d)
    return pd.DataFrame(prices_final)

def readVolatilities(fn1, fn2, fn3, CIKs):
    p1 = pd.read_csv(fn1, dtype={"datadate": int, "prccm": float, "cik": int})
    p1 = p1[['datadate','prccm','cik']]
    p2 = pd.read_csv(fn2, dtype={"datadate": int, "prccm": float, "cik": int})
    p2 = p2[['datadate','prccm','cik']]
    p3 = pd.read_csv(fn3, dtype={"datadate": int, "prccd": float, "cik": int})
    p3 = p3[['datadate','prccd','cik']]
    p3.columns = ['datadate','prccm','cik']
    prices = pd.concat([p1,p2,p3])
    del p1,p2,p3 
    volatility = []
    c = 0
    progress = 0.1
    #CIKs = ['0001000050']
    for cik in CIKs:
        print(cik)
        c+=1 
        if c/len(CIKs)>=progress:
            per = int(progress*100)
            print(str(per) + "%")
            progress+=0.1
        p_cik = prices.loc[prices['cik'] == int(cik)]
        mu = ((p_cik.sort_values('datadate'))['prccm'].pct_change()).mean()
        for year in range(2000,2019):
            low = (year-1)*10000+1001
            high = year*10000+931
            p = p_cik.loc[(p_cik['datadate'] >= low) & (p_cik['datadate'] <= high)]
            p = p[p['prccm'].notna()]
            q1 = p.loc[(p['datadate']//100%100>=10) & (p['datadate']//100%100<=12)]
            q2 = p.loc[(p['datadate']//100%100>=1) & (p['datadate']//100%100<=3)]
            q3 = p.loc[(p['datadate']//100%100>=4) & (p['datadate']//100%100<=6)]
            q4 = p.loc[(p['datadate']//100%100>=7) & (p['datadate']//100%100<=9)]
            year_str = str(year)
            qrt = [q1,q2,q3,q4]
            i=0
            for q in qrt:
                i+=1
                a = q.empty
                if q.empty == False:
                    _, ind = np.unique(q['datadate'].values, return_index=True)
                    q = q.iloc[list(ind)]
                    q = q.sort_values('datadate')
                    returns = q['prccm'].pct_change()
                    r1=r2=1
                    if len(returns[returns.notna()][returns[returns.notna()].ne(0)]):
                        r1 = returns[returns.notna()][returns[returns.notna()].ne(0)].iloc[0]
                        r2 = returns[returns.notna()][returns[returns.notna()].ne(0)].iloc[-1]
                    R_qrt = (r1-r2)/r1
                    eps_qrt = R_qrt - mu
                    mean_R = returns.mean()
                    vol_qrt = np.sqrt((np.square(returns - mean_R)).mean())
                    date = year_str+" Q"+str(i)
                    
                    n = math.floor(len(returns)/3)
                    returns2 = returns.iloc[0:n]
                    if len(returns2[returns2.notna()][returns2[returns2.notna()].ne(0)])>=2:
                        r1 = returns2[returns2.notna()][returns2[returns2.notna()].ne(0)].iloc[0]
                        r2 = returns2[returns2.notna()][returns2[returns2.notna()].ne(0)].iloc[-1]
                    R_mon = (r1-r2)/r1
                    eps_mon = R_mon - mu
                    vol_mon = np.sqrt((np.square(returns2 - returns2.mean())).mean())
                    v={'date':date,'vol_qrt':vol_qrt,'eps_qrt':eps_qrt,'vol_mon':vol_mon,'eps_mon':eps_mon,'cik':cik}
                    volatility.append(v)
    return pd.DataFrame(volatility)

def constructDataset():
    for year in range(2000,2019):
        print(year)
        f1 = loc+str(year)+"10X.pckl"
        f2 = "/Volumes/LaCie/Data/prices.pckl"
        f3 = "/Volumes/LaCie/Data/volatilities.pckl"
        final10X = fd.joinDatasets(f1,f2,f3)
        f4 = loc+str(year)+"10X_final.pckl"
        fd.saveFile(final10X, f4)
        del final10X
        
def constructDictionary():
    dictionary = defaultdict(dict)
    CIKs = []
    for year in range(2000,2014):
        print(year)
        filename = loc+str(year)+"10X_final.pckl"
        dictionary, cik = f10X.returnDictionary(dictionary, filename)
        CIKs+=cik
    dictionary = f10X.checkFrequency(dictionary)
    dictionary = f10X.checkDictionary(dictionary)
    return dictionary, CIKs

def getDescriptives():
    n10KP = 0
    n10KN = 0
    n10QP = 0
    n10QN = 0
    words10KP = 0
    words10KN = 0
    words10QP = 0
    words10QN = 0
    for year in range(2000,2019):
        print(year)
        filename = loc+str(year)+"10X_final.pckl"
        dataset = fd.loadFile(filename)
        for item in dataset:
            nWords = f10X.wordCount(item[5])
            text = open(item[2],"r").read()
            f_type = f10X.getFileType(text)
            y = item[4]
            if f_type == "10-K":
                if y >= 0:
                    n10KP+=1
                    words10KP+=nWords
                else:
                    n10KN+=1
                    words10KN+=nWords
            elif f_type == "10-Q":
                if y >= 0:
                    n10QP+=1
                    words10QP+=nWords
                else:
                    n10QN+=1
                    words10QN+=nWords
    descriptives = [n10KP, n10KN, n10QP, n10QN, words10KP, words10KN, words10QP, words10QN]
    row_names = ['# of positive 10Ks','# of negative 10Ks','# of positive 10Qs','# of negative 10Qs','# of words in positive 10Ks','# of words in negative 10Ks','# of words in positive 10Qs','# of words in negative 10Qs']
    return pd.DataFrame(descriptives, index = row_names)

#dataToList()
#constructDataset()
dictionary, CIKs = constructDictionary()
fd.saveFile(dictionary, loc+"dictionary_init.pckl")
#descriptives = getDescriptives()