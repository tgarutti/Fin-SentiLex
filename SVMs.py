##############################################################################
##### Support Vector Machines #################################################
##############################################################################
import numpy as np
import re
import pandas as pd
import functions10X as f10X
import functionsData as fd
import functionsNN as fNN
import functionsSVM as fSVM
import random as rd
import math
import collections
from collections import defaultdict
import time
from sklearn import svm
from sklearn import metrics

drive = '/Volumes/LaCie/Data/'
def svmDictionaries():
    loughranDict = fd.loadFile(drive+'Loughran_McDonald_dict.pckl')
    benchNNDict = fd.loadFile(drive+'dictionary_benchNN.pckl')
    classNNDict = fd.loadFile(drive+'dictionary_classificationNN.pckl')
    regresNNDict = fd.loadFile(drive+'dictionary_regressionNN.pckl')
    dictionaries = [benchNNDict, classNNDict, regresNNDict]
    
    dictionaries = fSVM.filterDicts(loughranDict, dictionaries, 0.4)
    return dictionaries

dictionaries = fd.loadFile(drive+'SVM_dictionaries.pckl')
dict_names = ['Loughran', 'Benchmark', 'Classification', 'Regression']

def SVMDataset(dictionaries, dict_names):
    train, test = dict(),dict()
    for d in dict_names:
        train[d] = []
        test[d] = []
    for year in range(2000,2015):
        print(year)
        filename = drive+str(year)+'10X_final.pckl'
        X = fSVM.getScores(filename, dictionaries, dict_names)
        for d in dict_names:
            train[d].extend(X[d])
    for year in range(2015,2019):
        print(year)
        filename = drive+str(year)+'10X_final.pckl'
        X = fSVM.getScores(filename, dictionaries, dict_names)
        for d in dict_names:
            test[d].extend(X[d])
    for name in dict_names:
        train[name] = np.row_stack(train[name])
        test[name] = np.row_stack(test[name])
    return train, test

train = fd.loadFile(drive+'train_final.pckl')
test = fd.loadFile( drive+'test_final.pckl')

start = time.time()
def forecastSVM(train, test, dict_names, ker, yValues):
    results = dict()
    for name in dict_names:
        results[name] = []
    for name in dict_names:
       for y in yValues: 
            X_train = train[name][:,2:13]
            X_train = fSVM.cleanMat(X_train.astype(np.float))
            X_train = fSVM.normalizeX(X_train)
            y_train = train[name][:,13:]
            y_train = y_train[:,y]
            
            X_test = test[name][:,2:13]
            X_test = fSVM.cleanMat(X_test.astype(np.float))
            X_test = fSVM.normalizeX(X_test)
            y_test = test[name][:,13:]
            y_test = y_test[:,y]
            
            X_train = X_train[:20000, 4:]
            y_train = y_train[:20000]
            X_test = X_test[:2000, 4:]
            y_test = y_test[:2000]
            
            X_train = np.delete(X_train, 4, 1)
            X_test = np.delete(X_test, 4, 1)
            X_train = np.delete(X_train, 1, 1)
            X_test = np.delete(X_test, 1, 1)
            
            X_train = fSVM.filterX(X_train)
            X_test = fSVM.filterX(X_test)
            if y==1 or y==2:
                
                clf = svm.NuSVR(gamma='auto', kernel = 'rbf')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results[name].append([y_pred, y_test])
            elif y==0 or y==3:
                f = fSVM.runSVM(X_train, y_train, X_test, y_test, 'sigmoid', 100, 0.01)
                results[name].append(f)
    return results

def resultsForecasts(forecasts, dict_names, yValues):
    results={}
    for y in yValues:
        res = []
        for name in dict_names:
            if y==0 or y==3:
                y_pred = forecasts[name][y][0].astype(np.int)
                y_true = forecasts[name][y][1].astype(np.int)
                accuracy = metrics.accuracy_score(y_true, y_pred)
                measures = fSVM.evaluationMeasures(y_true, y_pred, mean=False)
                res.append(measures)
            elif y==1 or y==2:
                y_pred = forecasts[name][y][0].astype(float)
                y_true = forecasts[name][y][1].astype(float)
                rmse = np.sqrt((np.square(y_pred - y_true)).mean())
                res.append(rmse)
        res = np.row_stack(res)
        resDF = pd.DataFrame(res)
        resDF.index = dict_names
        if len(res[0,:])==7:
            colNames = ['Precision (Pos.)', 'Precision (Neg.)', 'Recall (Pos.)', 'Recall (Neg.)', 'F1 (Pos.)', 'F1 (Neg.)', 'Accuracy']
            resDF.columns = colNames
        else:
            colNames = ['RMSE']
            resDF.columns = colNames
        results[y] = resDF
    return results


def subsampleSVM(train, test, dict_names, yValues, train_samples, test_samples, nSims):
    forecasts = defaultdict(dict)
    results = defaultdict(dict)
    for y in yValues:
        for i in range(len(train_samples)):
            train_len = train_samples[i]
            test_len = test_samples[i]
            fore = []
            res = []
            for j in range(nSims):
                dict_res = []
                for name in dict_names:
                    trainN = train[name][train[name][:, 0].argsort()[::-1]]
                    testN = test[name][test[name][:, 0].argsort()[::-1]]
                    nObs=int(0.2*len(trainN[:,0]))
                    trainN = trainN[:nObs,:]
                    np.random.shuffle(trainN)
                    np.random.shuffle(testN)
                    X_train, X_test, y_train, y_test = fSVM.getTrainTest(train, test, name, train_len, test_len, y)
                    X_train = fSVM.cleanMat(trainN[:train_len,4:8].astype(np.float))
                    X_test = fSVM.cleanMat(testN[:test_len,4:8].astype(np.float))
                    if y==0 or y==3:
                        y_pred, y_true = fSVM.runSVM(X_train, y_train, X_test, y_test, 'sigmoid', 100, 0.01)
                        measures = fSVM.evaluationMeasures(y_true, y_pred, mean=False)
                    elif y==1 or y==2:
                        y_pred, y_true = fSVM.runSVR(X_train, y_train, X_test, y_test, 'rbf', 0.001)
                        measures = np.sqrt((np.square(y_pred.astype(np.float) - y_true.astype(np.float))).mean())
                    fore.append([y_pred, y_true])
                    dict_res.append(measures)
                res.append(dict_res)
            if y==0 or y==3:
                R = np.row_stack(res)
                colnamesR = ['Precision (Pos.)', 'Precision (Neg.)', 'Recall (Pos.)', 'Recall (Neg.)', 'F1 (Pos.)', 'F1 (Neg.)', 'Accuracy']
                rownamesR = dict_names*nSims
            elif y==1 or y==2:
                R = np.column_stack(res)
                colnamesR = [train_len]*nSims
                rownamesR = dict_names
            Rdf = pd.DataFrame(R)
            Rdf.columns = colnamesR
            Rdf.index = rownamesR
            
            forecasts[y][train_len] = fore
            results[y][train_len] = Rdf
    return forecasts, results
            

#forecasts2,results2 = subsampleSVM(train, test, dict_names, [0,1,2,3], [2000,5000,10000,20000], [400,1000,2000,4000], 1)
#train, test = SVMDataset(dictionaries, dict_names)
forecasts = []
#forecasts = forecastSVM(train, test, dict_names, 'rbf', [0,1,2,3])
forecasts = fd.loadFile(drive+'forecasts.pckl')
#results = resultsForecasts(forecasts, dict_names, [0,1,2,3])
baseRet, baseVol = fSVM.getBaseline(train, test, [400,1000,2000,4000])

#gridSearch = gridSearch(train, test, dict_names, 0)
end = time.time()
print(end-start)