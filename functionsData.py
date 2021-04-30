##############################################################################
##### Functions for data loading/saving ######################################
##############################################################################       
import pickle
import pandas as pd
import numpy as np
import math
import functions10X as f10X
from collections import defaultdict


## Pickle: save and load file
def saveFile(file, filename):
    f = open(filename, 'wb')
    pickle.dump(file, f, protocol=-1)
    f.close()
    
def loadFile(filename):
    f = open(filename, 'rb')
    file = pickle.load(f)
    f.close()
    return file

## Write a list to txt file 
def listToText(listObj, filename):
    string = ''
    for i in listObj:
        string = string + str(i) + "\n"
    text = open(filename, 'w')
    text.write(string)
    text.close()
    
def readPrices(filename):
    prices = pd.read_csv(filename)
    prices = prices[~(prices['date']//100%100%3!=0)]
    for i in range(0,len(prices)):
        date = prices['date'].iloc[i]
        yr = date//10000
        cik = str(prices['cik'].iloc[i])
        prices['cik'].iloc[i] = "0"*(10-len(cik)) + cik
        if date//100%100//3==4:
            prices['date'].iloc[i] = str(yr) + " Q1"
        elif date//100%100//3==1:
            prices['date'].iloc[i] = str(yr) + " Q2"
        elif date//100%100//3==2:
            prices['date'].iloc[i] = str(yr) + " Q3"
        elif date//100%100//3==3:
            prices['date'].iloc[i] = str(yr) + " Q4"
    return prices.to_numpy()
    
def joinDatasets(f1, f2, f3):
    text10X = loadFile(f1)
    prices = loadFile(f2)
    prices = prices.to_numpy()
    vol = loadFile(f3)
    vol = vol.to_numpy()
    CIKs = np.unique(prices[:,2])
    text10Xfinal = []
    for item in text10X:
        date = item[0]
        date2 = incrementQuarter(date, 1)
        cik = item[1]
        if cik in CIKs:
            p_temp = prices[prices[:,2]==cik]
            v_temp = vol[vol[:,5]==cik]
            if date in p_temp[:,0] and date2 in p_temp[:,0]:
                p1 = p_temp[p_temp[:,0] == date][0,1]
                p2 = p_temp[p_temp[:,0] == date2][0,1]
                v_for = v_temp[v_temp[:,0] == date2][0,1:5]
                v_lag = []
                v_lag.append(v_temp[v_temp[:,0] == date][0,1:5])
                date3 = incrementQuarter(date, -1)
                if date3 in v_temp[:,0]:
                    v_lag.append(v_temp[v_temp[:,0] == date3][0,1:5])
                    date4 = incrementQuarter(date3, -1)
                    if date4 in v_temp[:,0]:
                        v_lag.append(v_temp[v_temp[:,0] == date4][0,1:5])
                if p1!=0:
                    p_change = (p2-p1)/p1
                    y = np.array([1,0])
                    y = np.array([0,1]) if p_change < 0 else y
                    text = open(item[2],"r").read()
                    f_type = f10X.getFileType(text)
                    item.insert(-1, f_type)
                    item.insert(-1, y)
                    item.insert(-1, p_change)
                    item.insert(-1, v_for)
                    item.insert(-1, v_lag)
                    text10Xfinal.append(item)
    return text10Xfinal

def loadTestDataset():
    loc = "/Users/user/Documents/Erasmus/QFMaster/Master Thesis/data_test/"
    dataset = loadFile(loc + "200010X_final.pckl")
    dictionary = loadFile(loc + "dictionary_2015.pckl")
    CIKs = loadFile(loc + "CIKs_final.pckl")
    prices = loadFile(loc + "prices.pckl")
    return dataset, dictionary, CIKs, prices

def incrementQuarter(date, increment):
    [yr, qrt] = date.split(" Q")
    yr = int(yr)
    qrt = int(qrt)
    if increment > 0:
        if qrt < 4:
            qrt = qrt+1
        else:
            yr = yr+1
            qrt = 1
    if increment < 0:
        if qrt > 1:
            qrt = qrt-1
        else:
            yr = yr-1
            qrt = 4
    return str(yr) + " Q" + str(qrt)

def cleanMatNP(mat):
    mat[mat==-np.inf] = 0
    mat[mat==np.inf] = 0
    mat[np.isnan(mat)] = 0
    return mat

def cleanMatPD(mat):
    mat[mat==-np.inf] = 0
    mat[mat==np.inf] = 0
    mat[pd.isnull(mat)] = 0
    return mat

def cleanMatPrices(mat):
    mat[mat==-np.inf] = 0
    mat[mat==np.inf] = 0
    mat[pd.isnull(mat)] = 0
    mat = mat[(mat[:,1].astype(np.float) > -1000) & (mat[:,1].astype(np.float) < 1000)]
    return mat

def cleanMatVol(mat):
    mat[mat==-np.inf] = 0
    mat[mat==np.inf] = 0
    mat[pd.isnull(mat)] = 0
    mat = mat[(mat[:,1].astype(np.float) > -10) & (mat[:,1].astype(np.float) < 10)]
    return mat

def descrNum(vec):
    mean = np.mean(vec)
    median = np.median(vec)
    std = np.std(vec)
    ma = np.max(vec)
    mi = np.min(vec)
    return [mean, median, std, ma, mi]

def slicer_vectorized(a,start,end):
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.fromstring(b.tostring(),dtype=(str,end-start))

def getQuantiles(mat):
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    results = []
    m = np.sort(mat[:,1]).astype(np.float)
    for p in quantiles:
        q = np.quantile(m,p)
        results.append(q)
    return results
        
def getPeriods(mat):
    periods = [2002, 2006, 2010, 2014, 2018]
    p0 = 1999
    results = []
    string = False
    if type(mat[:,0][0]) == np.str_:
        mat[:,0] = slicer_vectorized(mat[:,0].astype(str),0,4).astype(np.int)
        string = True
    
    for p in periods:
        if string == True:
            m = mat[(mat[:,0].astype(int)>p0) & (mat[:,0].astype(int)<=p)]
        else:
            m = mat[(mat[:,0]//10000>p0) & (mat[:,0]//10000<=p)]
        if len(m[:,1]) > 1:
            descr = descrNum(m[:,1].astype(np.float))
            descr.remove(descr[1])
            results = results + descr
        else:
            descr = [0,0,0,0]
            results = results + descr
        p0 = p
    return results

def getPercentageChange(mat,CIKs):
    results = []
    mat = mat[mat[:,0].argsort()]
    for cik in CIKs:
        m = pd.DataFrame(mat[mat[:,2]==cik])
        m[1] = (m[1].pct_change())
        m = m.iloc[1:]
        m = np.array(m)
        results.append(m)
    return np.row_stack(results)
        
        
def numericalDescriptives():
    drive = "/Volumes/LaCie/Data/"
    prices = loadFile(drive+'prices.pckl')
    vol = loadFile(drive+'volatilities.pckl')
    length = loadFile(drive+'length.pckl')
    prices=np.array(prices)
    CIKs = np.unique(prices[:,2])
    prices = cleanMatPrices(prices)
    prices = prices[prices[:,0].argsort()]
    change_price = loadFile(drive+'percentage_change.pckl')#getPercentageChange(prices, CIKs)
    saveFile(change_price, drive+'percentage_change.pckl')
    change_price = cleanMatVol(change_price)
    prices = prices[:,[0,1]]
    vol = np.array(vol)
    vol = vol[vol[:,0].argsort()]
    vol = cleanMatVol(vol)
    vol = vol[:,[0,3]]
    length[:,1] = length[:,1].astype(int)
    length = length[length[:,0].argsort()]
    length = cleanMatPD(length)
    change_price = cleanMatPD(change_price)
    numericals = [prices, change_price, vol, length]
    descriptives = []
    quantiles = []
    periods = []
    for mat in numericals:
        descriptives.append(descrNum(mat[:,1].astype(np.float)))
        quantiles.append(getQuantiles(mat))
        periods.append(getPeriods(mat))
    descriptives = pd.DataFrame(np.row_stack(descriptives))
    quantiles = pd.DataFrame(np.row_stack(quantiles))
    periods = pd.DataFrame(np.row_stack(periods))
    rownames = ['Stock Prices','Change in Stock Price','Stock Price Volatility','Document Length']
    colnames1 = ['Mean', 'Median', 'Std. Dev', 'Max.', 'Min.']
    colnames2 = ['0.05', '0.25', '0.5', '0.75', '0.95']
    colnames3 = ['Mean', 'Std. Dev', 'Max.', 'Min.', 'Mean', 'Std. Dev', \
                 'Max.', 'Min.', 'Mean', 'Std. Dev', 'Max.', 'Min.', 'Mean', \
                 'Std. Dev', 'Max.', 'Min.', 'Mean', 'Std. Dev', 'Max.', 'Min.']
    descriptives.columns = colnames1
    descriptives.index = rownames
    quantiles.columns = colnames2
    quantiles.index = rownames
    periods.columns = colnames3
    periods.index = rownames
    return descriptives, quantiles, periods
    
def ciksDescriptives(CIKs):
    drive = "/Volumes/LaCie/Data/"
    pd.options.mode.chained_assignment = None
    p1 = pd.read_csv(drive+'prices_crsp.csv', dtype={"datadate": int, "prccm": float, "cik": int})
    p1 = p1[['datadate','prccm','cik']]
    p2 = pd.read_csv(drive+'prices.csv', dtype={"datadate": int, "prccm": float, "cik": int})
    p2 = p2[['datadate','prccm','cik']]
    p3 = pd.read_csv(drive+'prices_daily.csv', dtype={"datadate": int, "prccd": float, "cik": int})
    p3 = p3[['datadate','prccd','cik']]
    p3.columns = ['datadate','prccm','cik']
    prices = pd.concat([p1,p2,p3])
    del p1,p2,p3
    length = loadFile(drive+'length.pckl')
    descriptives = defaultdict(dict)
    for cik in CIKs:
        p = prices[prices['cik']==int(cik)]
        l = length[length[:,2]==cik]
        p = p[p['prccm'].notna()]
        if len(p)>40 and len(l) > 20:
            returns = p.copy()
            returns['prccm']= (p['prccm'].pct_change())
            returns = returns.iloc[1:,:]
            volatility=[]
            for year in range(2000,2019):
                r_year = (returns.copy()).loc[(returns['datadate']//10000==year)]
                for mon in range(1,13):
                    r_mon = r_year.loc[(r_year['datadate']//100%100==mon)]
                    if len(r_mon)>1:
                        mean = np.mean(r_mon['prccm'])
                        r_mon.loc[:,'prccm'] = r_mon['prccm']-mean
                        vol = r_mon.copy()
                        vol.loc[:,'prccm'] = np.sqrt(np.mean(np.square(r_mon['prccm'])))
                        vol = vol.iloc[-1]
                        volatility.append(vol)
            volatility = np.row_stack(volatility)
            X = [np.array(p), np.array(returns), np.array(volatility), l]
            desc = []
            quantiles = []
            periods = []
            for mat in X:
                desc.append(descrNum(mat[:,1].astype(np.float)))
                quantiles.append(getQuantiles(mat))
                periods.append(getPeriods(mat))
            desc = pd.DataFrame(np.row_stack(desc))
            quantiles = pd.DataFrame(np.row_stack(quantiles))
            periods = pd.DataFrame(np.row_stack(periods))
            rownames = ['Stock Prices','Stock Return','Stock Return Volatility','Document Length']
            colnames1 = ['Mean', 'Median', 'Std. Dev', 'Max.', 'Min.']
            colnames2 = ['0.05', '0.25', '0.5', '0.75', '0.95']
            colnames3 = ['Mean', 'Std. Dev', 'Max.', 'Min.', 'Mean', 'Std. Dev', \
                         'Max.', 'Min.', 'Mean', 'Std. Dev', 'Max.', 'Min.', 'Mean', \
                         'Std. Dev', 'Max.', 'Min.', 'Mean', 'Std. Dev', 'Max.', 'Min.']
            desc.columns = colnames1
            desc.index = rownames
            quantiles.columns = colnames2
            quantiles.index = rownames
            periods.columns = colnames3
            periods.index = rownames
            descriptives[cik]['Descriptives'] = desc
            descriptives[cik]['Quantiles'] = quantiles
            descriptives[cik]['Periods'] = periods
    return descriptives

    
        
