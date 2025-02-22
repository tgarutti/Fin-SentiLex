##############################################################################
##### Benchmark Neural Network (Vo et al.) ###################################
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

def update(batch_dictDF, W, stepP, stepN, stepW, m, v):
    batch_dictDF.loc['pos'] = batch_dictDF.loc['pos']-stepP
    batch_dictDF.loc['neg'] = batch_dictDF.loc['neg']-stepN
    batch_dictDF.loc['mp'] = m[0]
    batch_dictDF.loc['mn'] = m[1]
    batch_dictDF.loc['vp'] = v[0]
    batch_dictDF.loc['vn'] = v[1]
    #W = W - np.diag(stepW)
    return batch_dictDF, W

def gradientW(d1, p1p2, X, w0, w1, batch_len):
    der0 = np.multiply(np.array([np.multiply(X[:,0],p1p2),np.multiply((-X[:,0]),p1p2)]),d1)
    der1 = np.multiply(np.array([np.multiply((-X[:,1]),p1p2),np.multiply(X[:,1],p1p2)]),d1)
    if w0 > 0:
        gradw0 = -der0.sum(axis=(0,1))/batch_len
    else:
        gradw0 = -der0.sum(axis=(0,1))/batch_len - 2*w0
    if w1 > 0:
        gradw1 = -der1.sum(axis=(0,1))/batch_len
    else:
        gradw1 = -der1.sum(axis=(0,1))/batch_len - 2*w1   
    return gradw0, gradw1

##### DIVIDE BY GRADIENTS BY LENGTH OF BATCH
def calculateGradients(y, y_hat, W, X, N):
    batch_len=len(y[0,:])
    w0 = W[0,0]
    w1 = W[1,1]
    sumX = X.sum(axis=0)
    d1 = np.divide(y,y_hat)
    p1p2 = np.multiply(y_hat[0,:],y_hat[1,:])
    
    # Gradients of word values
    derP = np.multiply(np.array([w0*p1p2,(-w0)*p1p2]),d1)
    derN = np.multiply(np.array([(-w1)*p1p2,w1*p1p2]),d1)
    gradP = -derP.sum(axis=(0))
    gradN = -derN.sum(axis=(0))    
    
    # Gradients for W
    gradw0 = gradw1 = 0.01
    #gradw0, gradw1 = gradientW(d1, p1p2, X, w0, w1, batch_len)
    
    grad = [gradP, gradN, gradw0, gradw1]
    
    return grad

def backPropagation(batch_dictDF, batch_mat, y, y_hat, W, X, m, v, N):
    grad = calculateGradients(y, y_hat, W, X, N)
    gradP = (grad[0]*batch_mat.T).sum(1)/len(y[0,:])
    gradN = (grad[1]*batch_mat.T).sum(1)/len(y[0,:])
    gradW = np.array([grad[2], grad[3]])
    stepP, mp, vp = fNN.adam(gradP, m[0], v[0])
    stepN, mn, vn = fNN.adam(gradN, m[1], v[1])
    stepW, mw, vw = fNN.adam(gradW, m[2], v[2])
    m = [mp,mn,mw]
    v = [vp,vn,vw]
    batch_dictDF, W = update(batch_dictDF, W, stepP, stepN, stepW, m, v)
    return batch_dictDF, W, m[2], v[2]

def forwardPropagation(batch, batch_dict, batch_mat, W):
    y = []
    y_hat = []
    X = []
    i = 0
    for item in batch:
        i+=1
        text = item[-1]
        price_boolean = item[4]
        price = item[5]
        y.append(price_boolean)
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
        
        # First linear layer - results in 2x1 vector
        linlayer = W.dot(sumValues)
        
        # Softmax
        y_hat.append(fNN.softmax(linlayer))
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

def initializeCoefficients():
    w1 = 0.1
    w2 = 0.1
    W = np.array([[w1,0],
                  [0,w2]])
    W = np.array([[w1,0],
                  [0,w2]])
    m = np.array([0,0])
    v = np.array([0,0])
    return W, m, v
    
def setHyperparameters():
    batch_size = 40
    epochs = 1
    return batch_size, epochs


def runNeuralNetwork(dataset, W, m_coef, v_coef):
#Initialize Neural Network
    for j in range(epochs):
        i, stop = fNN.newEpoch()
        while stop == False:
            batch, i, stop = fNN.nextBatch(dataset, i, batch_size, stop)
            batch_dict, batch_mat = batchDictionary(batch)
            euclid = fNN.euclideanNorm(batch_mat)
            batch_mat1 = batch_mat/euclid
            #batch_mat1 = fNN.tfidf(batch_mat)
            
            N = 0
            #N = (batch_mat.sum(1)).mean()
            #batch_mat1 = batch_mat/N
            batch_dictDF = pd.DataFrame(batch_dict)
            m = [batch_dictDF.loc['mp'],batch_dictDF.loc['mn'], m_coef]
            v = [batch_dictDF.loc['vp'],batch_dictDF.loc['vn'], v_coef]
            y, y_hat, X = forwardPropagation(batch, batch_dict, batch_mat1, W)
            loss.append(fNN.crossEntropyLoss(y, y_hat))
            batch_dictDF, W, m_coef, v_coef = backPropagation(batch_dictDF, batch_mat, y, y_hat, W, X, m, v, N)
            d = batch_dictDF.to_dict()
            dictionary.update(d)
            end2 = time.time()
    return loss, W
#dictionary = fd.loadFile(drive+'dictionary_final.pckl')
#dictionary = fNN.initializeX(dictionary)
dictionary = fd.loadFile(drive+'dictionary_benchNN.pckl')

W, m_coef, v_coef = initializeCoefficients()
batch_size, epochs = setHyperparameters()
loss = []
for year in range(2013,2015):
    start = time.time()
    dataset = fd.loadFile(drive+str(year)+'10X_final.pckl')
    rd.shuffle(dataset)
    loss, W = runNeuralNetwork(dataset, W, m_coef, v_coef)
    end = time.time()
    print(end-start)
fd.saveFile(dictionary, drive+'dictionary_benchNN.pckl')
