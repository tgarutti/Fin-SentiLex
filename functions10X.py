##############################################################################
##### Functions for reading and parsing a 10X file ###########################
##############################################################################
import numpy as np
import pandas as pd
import re           
import datetime
import string
import collections
import nltk
from nltk.corpus import stopwords
import random as rd
import functionsData as fd
from collections import defaultdict


##############################################################################
### GENERAL TEXT SEARCH FUNCTIONS ############################################
### General funcitons for searching string in text with error handling
    
# searchText - searches for first occurrence of text in file
def searchText(text, file):
    try:
        found = re.search(text, file).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = '' # apply error handling
    return found.strip()

# searchAll - searches for all occurences of text in file
def searchAll(text, file):
    try:
        found = re.findall(text, file, re.IGNORECASE)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = '' # apply error handling
    fin = []
    for s in found:
        fin.append(s.strip())
    return fin
##############################################################################
##############################################################################



##############################################################################
### READ HEADER ##############################################################
### Functions specific for reading the header of a 10X file

# readHeader - calls functions for the header and returns array of outcomes
def readHeader(text):
    sep = "</Header>"
    header, sep, tail = text.partition(sep)
    names = getNames(header)
    string_names = ", ".join(names)
    cik = getCIK(header)
    X = getFileType(header)
    return np.array([string_names, cik, X])

# getNames - returns current company names as well as former names
def getNames(text):
    names = []
    names.append(searchText('COMPANY CONFORMED NAME:(.+?)\n', text))
    name = searchAll('FORMER CONFORMED NAME:(.+?)\n', text)
    if name != '':
        names.extend(name)
    return names

# getCIK - returns the Central Index Key of the company
def getCIK(text):
    return searchText('CENTRAL INDEX KEY:(.+?)\n', text)

# getFileType - returns the file type (10-K, 10-Q, etc.)
def getFileType(text):
    return searchText('CONFORMED SUBMISSION TYPE:(.+?)\n', text)

# getDate - returns date, after a certain text, in datetime format  
def getDate(text, file):
    date = searchText(text, file)
    if date == '':
        return False
    else:
        return datetime.datetime.strptime(date, '%Y%m%d')

# checkDates - checks if two dates are further apart than the lag variable
def checkDates(text, lag):
    period = getDate('CONFORMED PERIOD OF REPORT:(.+?)\n', text)
    submission = getDate('FILED AS OF DATE:(.+?)\n', text)
    if period == False or submission == False:
        return False
    else:
        return submission-period <= datetime.timedelta(lag)
##############################################################################
##############################################################################



##############################################################################
### PARSING TEXT #############################################################
### Functions for cleaning and parsing text in separate lists per item

# itemize10X - split text item by item, returns list of item separated strings
def itemize10X(filename):
    text = open(filename,"r").read()
    text = partitionText(text, "</Header>", "</Ex>")
    items = checkItems(text)
    n = len(items)
    rem = text
    itemSep = []
    numVec = []
    itemizedText = ''
    for i in range(0,n):
        if i == n-1:
            item1 = ''.join(items[i])
            t = partitionText(rem, item1, '')
            rem = item1 + t
        else:
            item1 = ''.join(items[i])
            item2 = ''.join(items[i+1])
            t = partitionText(rem, item1, item2)
            rem = item2 + partitionText(rem, item2, '')
        if wordCount(t) > 0:
            num = [s for s in item1 if s.isdigit()]
            num = int(''.join(num))
            strItem = "ITEM " + str(num) + ".\n"
            
            if num in numVec:
                k = numVec.index(num)
                s_temp = itemSep[k]
                s_temp = s_temp + "\n\n" + t
                itemSep[k] = s_temp
            else:
                numVec.append(num)
                itemSep.append(strItem + t)    
    for j in itemSep:
        itemizedText = itemizedText + "\n\n" + j
    return itemizedText


# partitionText - returns the text partitioned by strings cut1 and cut2
# 1. if cut1 = '' -> return the text above cut2
# 2. if cut2 = '' -> return the text below cut2
# 3. if neither are 'null' -> return the text between cut1 and cut2
def partitionText(text, cut1, cut2):
    if cut1 == '':
        head, sep, tail = text.partition(cut2)
        final = head
    elif cut2 == '':
        head, sep, tail = text.partition(cut1)
        final = tail
    else:
        head, sep, tail = text.partition(cut1)
        head, sep, tail = tail.partition(cut2)
        final = head
    return final

# checkItems - returns list of 'Items' in text (if no items exist, empty list) 
def checkItems(text):
    itemSearch = re.findall("\n\s*(item)(\s*)([0-9]+)([:.!?\\-]|\s)(.*?)\n((.*?|\s*)*)(?:(?!item).)*", text, re.IGNORECASE)
    if not itemSearch:
        return itemSearch
    else:
        index = np.array(range(1,16))
        for item in itemSearch:
            if int(item[2]) not in index:
                itemSearch.remove(item)
        itemArray = np.array(itemSearch)
        if not itemSearch:
            return itemSearch
        else:
            itemNum = np.array(itemArray[:,2], dtype="int64")
            
            indItems = []
            maxInd = 0
            for i in index:
                a = np.where(itemNum == i)
                indItems.append(a)
                if np.size(a) > maxInd:
                    maxInd = np.size(a)
            return itemSearch

# wordCount - returns number of words in text
def wordCount(text):
    words = cleanText(text)
    return len(words)
    

# cleanText - seperate and clean text into a list of separate lowercased words
def cleanText(text):
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    cleanedWords = [s.lower() for s in stripped if s.lower().islower() and countNumbers(s)==0]
    return cleanedWords

# countNumbers - count number of digits in inputString
def countNumbers(inputString):
    n = 0
    for char in inputString:
        if char.isdigit():
            n = n+1
    return n

def checkDictionary(dictionary):
    nltk.download('words')
    nltk.download('stopwords')
    words = set(nltk.corpus.words.words())
    sw = set(stopwords.words('english'))
    delKeys = [key for key in dictionary.keys() if key not in words]
    for key in delKeys: 
        del dictionary[key]
    delKeys = [key for key in dictionary.keys() if key in sw]
    for key in delKeys: 
        del dictionary[key]
    return dictionary

def weightsDict(dictionary):
    NDOCS = 276880
    df = pd.DataFrame(dictionary)
    w1 = 1+np.log(df.loc['freq'])
    w2 = 1+np.log(df.loc['freq'].mean())
    w3 = np.log(NDOCS/df.loc['ndocs'])
    weights = (w1/w2)*w3
    return weights.sort_values()

def checkFrequency(dictionary):
    delKeys1 = [key for key in dictionary.keys() if dictionary[key]['freq'] < 100]
    delKeys2 = [key for key in dictionary.keys() if dictionary[key]['ndocs'] < 15000]
    delKeys3 = [key for key in dictionary.keys() if dictionary[key]['ndocs'] > 250000]
    delKeys = delKeys1+delKeys2+delKeys3
    for key in delKeys: 
        del dictionary[key]
    return dictionary


    

# constructDictionary - create dictionary
def returnDictionary(dictionary, filename):
    yearly_list = fd.loadFile(filename)
    CIKs =[]
    for k in yearly_list:
        CIKs.append(k[1])
        text = cleanText(k[-1])
        count = collections.Counter(text)
        for key, value in count.items():
            if key not in dictionary:
                dictionary[key]['pos'] = rd.randint(10, 50)/100
                dictionary[key]['neg'] = rd.randint(10, 50)/100
                dictionary[key]['mp'] = 0
                dictionary[key]['vp'] = 0
                dictionary[key]['mn'] = 0
                dictionary[key]['vn'] = 0
                dictionary[key]['freq'] = value
                dictionary[key]['ndocs'] = 1
            else:
                dictionary[key]['freq'] = dictionary[key]['freq'] + value
                dictionary[key]['ndocs'] += 1
    return dictionary, CIKs
##############################################################################
##############################################################################