##############################################################################
##### Functions for SVMs #####################################################
##############################################################################
import numpy as np
import re
import pandas as pd
import functions10X as f10X
import functionsData as fd
import functionsNN as fNN
import random as rd
import collections
from collections import defaultdict
import time
import math
from numpy.linalg import norm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors

def percentileDict(dictionary, percentile):
    sort = fNN.getSortedScores(dictionary)
    quant = ((sort.abs()).sort_values(ascending=False)).quantile(percentile)
    df_filtered = sort.drop(sort[(sort >= -quant) & (sort <= quant)].index)
    return df_filtered.to_dict()

def filterDicts(loughranDict, dictionaries, percentile):
    for i in range(len(dictionaries)):
        dictionaries[i] = percentileDict(dictionaries[i], percentile)
    dictionaries.insert(0, loughranDict)
    return dictionaries

def filterDicts(loughranDict, dictionaries, percentile):
    loughranDict =  pd.DataFrame(loughranDict, index=['Score']).to_dict()
    for i in range(len(dictionaries)):
        if i == 0:
            scores = pd.DataFrame(percentileDict(dictionaries[i], percentile), index=['Score'])
            dictionaries[i] = scores.to_dict()
        else:
            dictDF = pd.DataFrame(dictionaries[i])
            scores = pd.DataFrame(percentileDict(dictionaries[i], percentile), index=['Score'])
            dictDF = dictDF.drop(['pos','neg','mp','vp','mn','vn','freq'])
            i1 = scores.columns.tolist()
            i2 = dictDF.columns.tolist()
            iRemove = list(set(i2).difference(set(i1)))
            dictDF = dictDF.drop(columns=iRemove)
            d = dictDF.to_dict() 
            for word in i1:
                d[word]['Score'] = scores[word].loc['Score']
            dictionaries[i] = d
    dictionaries.insert(0, loughranDict)
    return dictionaries

def nPos(A):
    array = A.copy()
    array[array<0] = 0
    array[array>0] = 1
    return int(sum(array))

def nNeg(A):
    array = A.copy()
    array[array>0] = 0
    array[array<0] = 1
    return int(sum(array))

def getOmega(text, dictionary):
    text = f10X.cleanText(text)
    values = [dictionary[w]['Score'] for w in text if w in dictionary]
    if len(values)==0:
        return []
    else:  
        len_text = len(text)
        n_sentWords = len(values)
        values = np.array(values)
        n_pos = nPos(values)
        n_neg = nNeg(values)
        score1 = int(sum(values))
        #scorw2 
        return [len_text, n_sentWords, n_pos, n_neg, score1]
    
def getOmega2(text, dictionary):
    N = 276880
    freq = collections.Counter(f10X.cleanText(text))
    avg_count = sum(freq.values())/len(freq.values())
    values = []
    for key, value in freq.items():
        if key in dictionary:
            score = dictionary[key]['Score']
            weight = (1+np.log(value))/(1+np.log(avg_count))*np.log(dictionary[key]['ndocs']/N)
            values.append(score*weight)
    values = [dictionary[w]['Score'] for w in text if w in dictionary]
    if len(values)==0:
        return []
    else:  
        len_text = len(text)
        n_sentWords = len(values)
        values = np.array(values)
        n_pos = nPos(values)
        n_neg = nNeg(values)
        score1 = int(sum(values))
        #scorw2 
        return [len_text, n_sentWords, n_pos, n_neg, score1]

def getScores(filename, dictionaries, dict_names):
    dataset = fd.loadFile(filename)
    data = dict()
    for d in dict_names:
        data[d] = []
    for item in dataset:
        x1 = np.concatenate(np.row_stack(item[7])[:,0:2]).tolist()
        n = 6-len(x1)
        for i in range(n):
            x1.insert(0, 0)
        if item[7][0][0] != 0:
            vol_change = (item[6][0]-item[7][0][0])/item[7][0][0]
            if vol_change >=0:
                y = [item[4][0],item[5],vol_change,1]
            else:
                y = [item[4][0],item[5],vol_change,0]
    
            for i in range(len(dictionaries)):
                d = dictionaries[i]
                d_name = dict_names[i]
                if d_name == 'Classification' or d_name == 'Regression':
                    x2 = getOmega2(item[-1], d)
                else:
                    x2 = getOmega(item[-1], d)
                if len(x2) > 0:
                    info = [item[0], item[1]]
                    x = np.array(info+x1+x2+y)
                    data[d_name].append(x)
    return data

def normalizeX(X):
    n = []
    for i in range(len(X[0,:])):
        n.append(norm(X[:,i]))
    n = np.array(n)
    n[0:6]=1
    Xnew = X
    Xnew[:,6] = np.ceil(X[:,6]/10000)
    Xnew[:,7:] = np.ceil(X[:,7:]/100)
    return Xnew

def deMean(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    return (vec-mean)/std

def filterX(X):
    Xnew = X
    Xnew = np.delete(Xnew, 3, 1)
    for i in range(len(Xnew[0,:])):
        if i == 2:
            Xnew[:,i] = X[:,3]/X[:,2]
        else:
            Xnew[:,i] = deMean(Xnew[:,i])
    return Xnew

def cleanMat(mat):
    mat[mat==-np.inf] = 0
    mat[mat==np.inf] = 0
    mat[np.isnan(mat)] = 0
    return mat

def formatDataset(train, test, name, y, train_len, test_len):
    X_train = train[name][:,2:13]
    X_train = cleanMat(X_train.astype(np.float))
    X_train = normalizeX(X_train)
    y_train = train[name][:,13:]
    y_train = y_train[:,y]
    
    X_test = test[name][:,2:13]
    X_test = cleanMat(X_test.astype(np.float))
    X_test = normalizeX(X_test)
    y_test = test[name][:,13:]
    y_test = y_test[:,y]
    
    X_train = X_train[:train_len, 4:]
    y_train = y_train[:train_len]
    X_test = X_test[:test_len, 4:]
    y_test = y_test[:test_len]
    
    X_train = np.delete(X_train, 4, 1)
    X_test = np.delete(X_test, 4, 1)
    X_train = np.delete(X_train, 1, 1)
    X_test = np.delete(X_test, 1, 1)
    
    X_train = filterX(X_train)
    X_test = filterX(X_test)
    return X_train, y_train, X_test, y_test

def runSVM(X_train, y_train, X_test, y_test, k, c, g):
    clf = svm.SVC(C = c, gamma=g, kernel = k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test

def runSVR(X_train, y_train, X_test, y_test, k, g):
    clf = svm.NuSVR(gamma=g, kernel = k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test

    
def confusionMatrix(y_true, y_pred):
    mat = metrics.confusion_matrix(y_true, y_pred)
    return mat

def evaluationMeasures(y_true, y_pred, mean):
    cm = confusionMatrix(y_true, y_pred)
    precision1 = cm[0,0]/(cm[0,0]+cm[1,0])
    precision2 = cm[1,1]/(cm[1,1]+cm[0,1])
    pos = cm[0,0]/(cm[0,0]+cm[0,1])
    pos = cm[1,1]/(cm[1,0]+cm[1,1])
    recall1 = cm[0,0]/(cm[0,0]+cm[0,1])
    recall2 = cm[1,1]/(cm[1,1]+cm[1,0])
    f1 = 2*(recall1 * precision1) / (recall1 + precision1)
    f2 = 2*(recall2 * precision2) / (recall2 + precision2)
    accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    if mean == True:
        precision = (precision1+precision2)/2
        recall = (recall1+recall2)/2
        f = 2*(recall * precision) / (recall + precision)
        return [precision, recall, f, accuracy]
    else:
        return [precision1, precision2, recall1, recall2, f1, f2, accuracy]
    
def getBaseline(train, test, sampleSizes):
    y_vol = test['Loughran'][:,-1].astype(float)
    y_ret = test['Loughran'][:,-4].astype(float)
    train_vol = train['Loughran'][:,-1].astype(float)
    train_ret = train['Loughran'][:,-4].astype(float)
    metrics_ret = []
    metrics_vol = []
    for s in sampleSizes:
        volBool = False
        while volBool==False:
            np.random.shuffle(y_vol)
            np.random.shuffle(train_vol)
            y_volSample = y_vol[:s]
            s2 = s*4
            train_volSample = train_vol[:s2]
            if sum(train_volSample)/len(train_volSample) >= 0.5:
                y_volPred = np.zeros(len(y_volSample))+1
            else:
                y_volPred = np.zeros(len(y_volSample))
            y_volMetrics = evaluationMeasures(y_volSample, y_volPred, False)
            if y_volMetrics[-1] < 0.52:
                volBool = True
        metrics_vol.append(y_volMetrics)
        
        retBool = False
        while retBool==False:
            np.random.shuffle(y_ret)
            np.random.shuffle(train_ret)
            y_retSample = y_ret[:s]
            s2 = s*4
            train_retSample = train_ret[:s2]
            if sum(train_retSample)/len(train_retSample) >= 0.5:
                y_retPred = np.zeros(len(y_retSample))+1
            else:
                y_retPred = np.zeros(len(y_retSample))
            y_retMetrics = evaluationMeasures(y_retSample, y_retPred, False)
            if y_retMetrics[-1] < 0.52:
                retBool = True
        metrics_ret.append(y_retMetrics)
    return np.array(metrics_ret), np.array(metrics_vol)

def getTrainTest(train, test, name, n_train, n_test, y):
    X_train = train[name][:,2:13]
    X_train = cleanMat(X_train.astype(np.float))
    X_train = normalizeX(X_train)
    y_train = train[name][:,13:]
    y_train = y_train[:,y]
    
    X_test = test[name][:,2:13]
    X_test = cleanMat(X_test.astype(np.float))
    X_test = normalizeX(X_test)
    y_test = test[name][:,13:]
    y_test = y_test[:,y]
    
    X_train = X_train[(-n_train):, 4:]
    y_train = y_train[(-n_train):]
    X_test = X_test[:n_test, 4:]
    y_test = y_test[:n_test]
    
    X_train = np.delete(X_train, 4, 1)
    X_test = np.delete(X_test, 4, 1)
    X_train = np.delete(X_train, 1, 1)
    X_test = np.delete(X_test, 1, 1)
    
    X_train = filterX(X_train)
    X_test = filterX(X_test)
    return X_train, X_test, y_train, y_test
                
def manualGridSearch(train, test, dict_names, y, X, Y, tuning_parameters):
    gridSearch = defaultdict(dict)
    for i in range(len(X)):
        nX = X[i]
        nY = Y[i]
        for k in tuning_parameters['kernel']:
            d = 3
            print(k)
            listK = []
            if k =='poly':
                d = int(k[-1])
                k = 'poly'
            for g in tuning_parameters['gamma']:
                print("   "+str(g))
                col = []
                if 'poly' in k and type(g) != float:
                    for c in tuning_parameters['C']:
                        for name in dict_names:
                            col.append(0)
                else:
                    for c in tuning_parameters['C']:
                        if 'poly' in k:
                            k = 'poly'
                        print("      "+str(c))
                        for name in dict_names:
                            print("      "+name)
                            X_train, X_test, y_train, y_test = getTrainTest(train, test, name, nX, nY, y)
                            if y==1 or y==2:
                                clf = svm.NuSVR(kernel=k, C=c, gamma=g, degree=d)
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                y_pred = y_pred.astype(np.float)
                                y_test = y_test.astype(np.float)
                                rmse = np.sqrt((np.square(y_pred - y_test)).mean())
                                col.append(rmse)
                            elif y==0 or y==3:
                                clf = svm.SVC(kernel=k, C=c, gamma=g, degree=d)
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                y_pred = y_pred.astype(np.int)
                                y_test = y_test.astype(np.int)
                                precision = metrics.precision_score(y_true=y_test,y_pred=y_pred,pos_label=0)
                                col.append(precision)
                listK.append(col)
            A = np.column_stack(listK)
            row_names = len(tuning_parameters['C'])*dict_names
            col_names = tuning_parameters['gamma']
            A = pd.DataFrame(A)
            A.index = row_names
            A.columns = col_names
            gridSearch[k][nX] = A
    return gridSearch

def gridSearch(train, test, dict_names, y):
    tuning_parameters = {'kernel': ['rbf','poly2','poly3','sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 'auto'], 'C': [1,10,100]}
    gridSearch = manualGridSearch(train, test, dict_names, 3, [2000,5000,10000], [400,1000,2000], tuning_parameters)
    return gridSearch

def plotGrid(data1, data2, name):

    fig, (ax1, ax2) = plt.subplots(1,2)
    x = data1[0]
    y = data1[1]
    z = data1[2]
    ax1.scatter(x, y, c=z, s=50, cmap=mpl.cm.Blues)
    xlabels = [item.get_text() for item in ax1.get_xticklabels()]
    xlabels[1] = 'Linear'
    xlabels[2] = 'Poly(3)'
    xlabels[3] = 'RBF'
    xlabels[4] = 'Sigmoid'
    ylabels = [item.get_text() for item in ax1.get_yticklabels()]
    ylabels[1] = '1e-2'
    ylabels[2] = '1e-3'
    ylabels[3] = '1e-4'
    ylabels[4] = '1e-5'
    ylabels[5] = 'Auto'
    ylabels[6] = 'Scale'
    ax1.set_xticklabels(xlabels)
    ax1.set_yticklabels(ylabels)
    ax1.set(xlabel='Kernel',ylabel='Gamma')
    ax1.set_title('Classification Lexicon')
    
    
    x = data2[0]
    y = data2[1]
    z = data2[2]
    ax2.scatter(x, y, c=z, s=50, cmap=mpl.cm.Blues)
    xlabels = [item.get_text() for item in ax2.get_xticklabels()]
    xlabels[1] = 'Linear'
    xlabels[2] = 'Poly(3)'
    xlabels[3] = 'RBF'
    xlabels[4] = 'Sigmoid'
    ylabels = [item.get_text() for item in ax2.get_yticklabels()]
    ylabels[1] = '1e-2'
    ylabels[2] = '1e-3'
    ylabels[3] = '1e-4'
    ylabels[4] = '1e-5'
    ylabels[5] = 'Auto'
    ylabels[6] = 'Scale'
    ax2.set_xticklabels(xlabels)
    ax2.set_yticklabels(ylabels)
    ax2.set(xlabel='Kernel')
    ax2.set_title('Regression Lexicon')
    cNorm = colors.Normalize(vmin=0.25, vmax=0.65)
    fig.colorbar(cm.ScalarMappable(norm = cNorm, cmap=mpl.cm.Blues), ticks=[0.3,0.4,0.5,0.6], label='Accuracy')
    #cNorm = colors.Normalize(vmin=0.18, vmax=0.38)
    #clb = fig.colorbar(cm.ScalarMappable(norm = cNorm, cmap=mpl.cm.Blues), label='RMSE')
    #clb.set_ticks([0.2,0.25,0.3,0.35])
    #clb.set_ticklabels([0.2,0.25,0.3,0.35][::-1])
    plt.savefig(name+'.eps')
    plt.show()