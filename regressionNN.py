##############################################################################
##### Regression Neural Network ##########################################
##############################################################################
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
test_loc = '/Users/user/Documents/Erasmus/QFMaster/Master Thesis/data_test/'

def update(batch_dictDF, coefficients, stepP, stepN, stepCoef, m, v):
    batch_dictDF.loc['pos'] = batch_dictDF.loc['pos']-stepP
    batch_dictDF.loc['neg'] = batch_dictDF.loc['neg']-stepN
    batch_dictDF.loc['mp'] = m[0]
    batch_dictDF.loc['mn'] = m[1]
    batch_dictDF.loc['vp'] = v[0]
    batch_dictDF.loc['vn'] = v[1]
    coefficients[1] = coefficients[1] - np.diag(stepCoef[0])
    coefficients[2] = coefficients[2] - np.diag(stepCoef[1])
    coefficients[3] = coefficients[3] - stepCoef[2]
    return batch_dictDF, coefficients

def gradientsCoef(coefficients, y, y_hat, X, batch_len):
    batch_len=len(y[0,:])
    D = coefficients[1]
    W = coefficients[2]
    
    h = X
    a = h.dot(D)
    e2a = np.exp(2*a)
    z = (e2a-1)/(e2a+1)
    b = W[0]*z[0] - W[1]*z[1]
    
    d1 = 2(y-y_hat)
    
    gradw0 = np.multiply(d1,z[:,0]).sum(axis=(0,1))/batch_len
    gradw1 = np.multiply(d1,-z[:,1]).sum(axis=(0,1))/batch_len
    
    # e2h = np.exp(2*h)
    # sech = (2*np.exp(h))/(e2h + 1)
    
    a2 = a 
    a2[(a2 <=1) & (a2 >= -1)] = 1
    a2[(a2 > 1) | (a2 < -1)] = 0
    
    gradd0 = np.multiply(d1,W[0,0]*np.multiply(a2,h[:,0]).T).sum(axis=(0,1))/batch_len
    gradd1 = np.multiply(d1,(-W[1,1])*np.multiply(a2,h[:,1]).T).sum(axis=(0,1))/batch_len
    
    gradCoefs = [np.array([gradd0, gradd1]), np.array([gradw0, gradw1])]
    
    return gradCoefs

##### DIVIDE BY GRADIENTS BY LENGTH OF BATCH
def calculateGradients(y, y_hat, coefficients, X, N):
    batch_len=len(y[0,:])
    betas = coefficients[0]
    D = coefficients[1]
    W = coefficients[2]
    
    h = X
    h = X
    a = h.dot(D)
    a2 = a 
    a2[(a2 <=1) & (a2 >= -1)] = 1
    a2[(a2 > 1) | (a2 < -1)] = 0
    e2a = np.exp(2*a)
    z = (e2a-1)/(e2a+1)
    b = W[0]*z[0] - W[1]*z[1]
    
    d1 = 2(y-y_hat)
    
    # Gradients of word values
    pos = W[0]*a2*D[0,0]
    neg = (-W[1])*a2*D[1,1]
    derP = np.multiply(pos,d1)
    derN = np.multiply(neg,d1)
    gradP = -derP.sum(axis=(0))
    gradN = -derN.sum(axis=(0))    
    
    # Gradients for Coef
    gradCoefs = gradientsCoef(coefficients, y, y_hat, X, batch_len)
    
    grad = [gradP, gradN]
    
    return grad, gradCoefs

def backPropagation(batch_dictDF, batch_mat, y, y_hat, coefficients, X, m, v, N):
    # grad, gradCoefs = calculateGradients(y, y_hat, coefficients, X, N)
    # gradP = (grad[0]*batch_mat.T).sum(1)/len(y[0,:])
    # gradN = (grad[1]*batch_mat.T).sum(1)/len(y[0,:])
    # stepP, mp, vp = fNN.adam(gradP, m[0], v[0])
    # stepN, mn, vn = fNN.adam(gradN, m[1], v[1])
    # m=[mp,mn,0]
    # v=[vp,vn,0]
    # stepCoef = 0
    # batch_dictDF, W = update(batch_dictDF, coefficients, stepP, stepN, stepCoef, m, v)
    grad, gradCoefs = calculateGradients(y, y_hat, coefficients, X, N)
    gradP = (grad[0]*batch_mat.T).sum(1)/len(y[0,:])
    gradN = (grad[1]*batch_mat.T).sum(1)/len(y[0,:])
    stepP, mp, vp = fNN.adam(gradP, m[0], v[0])
    stepN, mn, vn = fNN.adam(gradN, m[1], v[1])
    mCoefs = m[2]
    vCoefs = v[2]
    stepD, mCoefs[1], vCoefs[1] = fNN.adam(gradCoefs[0], mCoefs[1], vCoefs[1])
    stepW, mCoefs[2], vCoefs[2] = fNN.adam(gradCoefs[1], mCoefs[2], vCoefs[2])
    stepC, mCoefs[3], vCoefs[3] = fNN.adam(gradCoefs[2], mCoefs[3], vCoefs[3])
    stepCoef = [stepD, stepW, stepC] #Change to [0, 0, 0] to not update coefficients
    m=[mp,mn,mCoefs]
    v=[vp,vn,vCoefs]
    batch_dictDF, coefficients = update(batch_dictDF, coefficients, stepP, stepN, stepCoef, m, v)
    return batch_dictDF, coefficients, m[2], v[2]

def forwardPropagation(batch, batch_dict, batch_mat, coefficients):
    betas = coefficients[0]
    D = coefficients[1]
    W = coefficients[2]
    y = []
    y_hat = []
    X = []
    i = 0
    for item in batch:
        i+=1
        text = item[-1]
        price_boolean = item[4]
        price = item[5]
        y.append(price)
        text = f10X.cleanText(text)
        text = list(set(text))        
        # From text to values
        doc = "doc"+str(i)
        values = [[batch_mat.at[doc,w]*batch_dict[w]['pos'],batch_mat.at[doc,w]*batch_dict[w]['neg']] for w in text if w in batch_dict]
        #values = [[batch_dict[w]['pos'],batch_dict[w]['neg']] for w in text if w in batch_dict]
        values = np.array(values)
                
        # Summation layer - results in 2x1 vector
        sumValues = (values.sum(axis=0))
        X.append(sumValues)
        
        # NN layers
        linlayer1 = D.dot(sumValues)
        e2a = np.exp(2*linlayer1)
        tanh = (e2a-1)/(e2a+1)
        
        y_hat.append(W[0]*tanh[0] - W[1]*tanh[1])
    y = np.column_stack(y)
    y_hat = np.column_stack(y_hat)
    X = np.row_stack(X)
    return y, y_hat, X
    
def batchDictionary(batch):
    fullTexts = ''
    fullTexts = ''.join([fullTexts + doc[-1] for doc in batch])
    inter = list(set(f10X.cleanText(fullTexts)).intersection(dictionary.keys()))
    colNames = inter
    rowNames = ["doc"+str(i+1) for i in range(len(batch))]
    zeros = np.zeros((len(rowNames),len(colNames)))
    batch_mat = pd.DataFrame(zeros, index=rowNames, columns=colNames)
    batch_dict = {}
    i=1
    for doc in batch:
        rowStr = "doc"+str(i)
        docList = list(set(f10X.cleanText(doc[-1])).intersection(inter))
        d = {}
        d = {k: dictionary[k] for k in docList}
        freq = collections.Counter(f10X.cleanText(doc[-1]))
        for k in docList:
            rowStr = "doc"+str(i)
            batch_mat.loc[rowStr,k] = freq[k]
        batch_dict.update(d)
        i+=1
    return batch_dict, batch_mat

def itemizedBatchDictionary(batch, betas):
    fullTexts = ''
    fullTexts = ''.join([fullTexts + doc[-1] for doc in batch])
    intersect = list(set(f10X.cleanText(fullTexts)).intersection(dictionary.keys()))
    colNames = intersect
    rowNames = ["doc"+str(i+1) for i in range(len(batch))]
    zeros = np.zeros((len(rowNames),len(colNames)))
    batch_mat = pd.DataFrame(zeros, index=rowNames, columns=colNames)
    betas_mat = pd.DataFrame(zeros, index=rowNames, columns=colNames)
    Xs_mat = pd.DataFrame(zeros, index=rowNames, columns=colNames)
    XtimesBeta = pd.DataFrame(zeros, index=rowNames, columns=colNames)
    batch_dict = {}
    i=1
    for doc in batch:
        rowStr = "doc"+str(i)
        docList = list(set(f10X.cleanText(doc[-1])).intersection(intersect))
        d = {}
        d = {k: dictionary[k] for k in docList}
        freq = collections.Counter(f10X.cleanText(doc[-1]))
        items = fNN.getItems(doc[-1])
        for item in items:
            itemList = list(set(f10X.cleanText(item[-1])).intersection(docList))
        for k in docList:
            rowStr = "doc"+str(i)
            n = dictionary[k]['ndocs']
            batch_mat.loc[rowStr,k] = (1+np.log(freq[k]))*np.log(n_docs/n)
        batch_dict.update(d)
        i+=1
    return batch_dict, batch_mat

def initializeCoefficients():
    W = np.array([1,1])
    m_w = np.array([0,0])
    v_w = np.array([0,0])
    
    D = np.array([[0.01,0],
                  [0,0.01]])
    m_d = np.array([0,0])
    v_d = np.array([0,0])
    
    B = 0.1*np.ones((1,15))
    m_b = np.zeros((1,15))
    v_b = np.zeros((1,15))
    
    coefficients = [B,D,W]
    Ms = [m_b, m_d, m_w]
    Vs = [v_b, v_d, v_w]
    return coefficients, Ms, Vs
    
def setHyperparameters():
    batch_size = 40
    epochs = 1
    return batch_size, epochs


def runNeuralNetwork(dataset, coefficients, Ms, Vs):
#Initialize Neural Network
    for j in range(epochs):
        i, stop = fNN.newEpoch()
        while stop == False:
            batch, i, stop = fNN.nextBatch(dataset, i, batch_size, stop)
            batch_dict, batch_mat = batchDictionary(batch)
            #euclid = fNN.euclideanNorm(batch_mat)
            #batch_mat1 = batch_mat/euclid
            N = 0
            #N = (batch_mat.sum(1)).mean()
            #batch_mat1 = batch_mat/N
            batch_mat1 = fNN.tfidf2(batch_mat)
            batch_dictDF = pd.DataFrame(batch_dict)
            m = [batch_dictDF.loc['mp'],batch_dictDF.loc['mn'], Ms]
            v = [batch_dictDF.loc['vp'],batch_dictDF.loc['vn'], Vs]
            y, y_hat, X = forwardPropagation(batch, batch_dict, batch_mat1, coefficients)
            loss.append(fNN.MSELoss(y, y_hat))
            batch_dictDF, coefficients, Ms, Vs = backPropagation(batch_dictDF, batch_mat, y, y_hat, coefficients, X, m, v, N)
            d = batch_dictDF.to_dict()
            dictionary.update(d)
            end2 = time.time()
    return loss, coefficients
time.sleep(15000)
dictionary = fd.loadFile(drive+'dictionary_filtered.pckl')
dictionary = fNN.initializeX(dictionary)
#dictionary = fd.loadFile(drive+'dictionary_regressionNN.pckl')
n_docs = 276880

coefficients, Ms, Vs = initializeCoefficients()
batch_size, epochs = setHyperparameters()
loss = []
for year in range(2013,2015):
    start = time.time()
    dataset = fd.loadFile(drive+str(year)+'10X_final.pckl')
    rd.shuffle(dataset)
    loss, coefficients = runNeuralNetwork(dataset, coefficients, Ms, Vs)
    end = time.time()
    print(end-start)
fd.saveFile(dictionary, drive+'dictionary_regressionNN.pckl')
