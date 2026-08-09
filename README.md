# Fin-SentiLex

Implementation for the automatic construction and evaluation of finance-specific sentiment lexicons using SEC 10-X filings.

The project builds sentiment dictionaries from financial disclosures and evaluates them using dictionary-based scores, Support Vector Machines, and custom neural-network models. The main goal is to compare manually curated and automatically generated financial sentiment lexicons in terms of their ability to explain or predict financial outcomes.

## Overview

Fin-SentiLex implements a research pipeline for:

1. Collecting and parsing SEC 10-X filings.
2. Building a financial text corpus.
3. Constructing sentiment dictionaries with positive and negative word scores.
4. Training dictionary-based neural networks to update word sentiment scores.
5. Evaluating the dictionaries through SVM/SVR models.
6. Benchmarking different dictionary construction methods.

The models are designed around the idea that each document can be represented by aggregated positive and negative sentiment scores computed from the words appearing in the filing.

## Repository Structure

```text
Fin-SentiLex/
│
├── dictionaries/              # Sentiment dictionaries and generated lexicons
│
├── get10Xlist.py              # Builds the list of SEC 10-X filings
├── getFileList10X.py          # Creates file lists for downloaded filings
├── searchFiles.py             # Searches and retrieves filing files
├── functions10X.py            # Utilities for cleaning and parsing 10-X text
│
├── parsingData.py             # Data parsing and preprocessing
├── functionsData.py           # File loading/saving and data utilities
│
├── SVMs.py                    # SVM/SVR evaluation of sentiment dictionaries
├── functionsSVM.py            # Helper functions for SVM scoring and evaluation
│
├── classificationNN.py        # Neural network for binary classification
├── regressionNN.py            # Neural network for continuous-target regression
├── functionsNN.py             # Neural-network utility functions
│
├── benchmarkDict.py           # Benchmarking of dictionary-based approaches
├── benchmarkNN.py             # Benchmarking of neural-network dictionaries
│
└── README.md
```

# Methodology

## Overview

Fin-SentiLex is a framework for the automatic construction and evaluation of finance-specific sentiment lexicons using SEC 10-X filings.

The framework consists of three main stages:

1. Text collection and preprocessing.
2. Dictionary construction using neural networks.
3. Dictionary evaluation using SVM and SVR models.

The workflow is shown below:

```text
SEC Filings
      ↓
Text Processing
      ↓
Initial Dictionary
      ↓
Classification NN
      ↓
Classification Dictionary
      ↓
Regression NN
      ↓
Regression Dictionary
      ↓
SVM / SVR Evaluation
      ↓
Performance Comparison
```

---

# Data

The framework uses annual SEC 10-X filings stored as serialized Python objects.

Each filing contains:

| Field | Description |
|---------|-------------|
| `item[4]` | Binary classification target |
| `item[5]` | Continuous regression target |
| `item[-1]` | Filing text |

The datasets are organized by year and stored as:

```text
200010X_final.pckl
200110X_final.pckl
...
201810X_final.pckl
```

---

# Text Processing

Filing text is cleaned and tokenized using utilities in `functions10X.py`.

For every document:

1. Text is cleaned and normalized.
2. Tokens are extracted.
3. Only words appearing in the sentiment dictionary are retained.
4. Document-word matrices are constructed.
5. TF-IDF weighting is applied.

Each document is ultimately represented using aggregated positive and negative sentiment scores.

---

# Sentiment Dictionary

The sentiment dictionary assigns two values to each word:

```python
dictionary[word]['pos']
dictionary[word]['neg']
```

representing positive and negative sentiment strength.

Additional fields are stored for optimization and document statistics, but the neural networks primarily learn the positive and negative sentiment values.

---

# Document Representation

For each document, all dictionary words are converted into sentiment vectors:

```text
[word_positive_score,
 word_negative_score]
```

These values are weighted using TF-IDF and aggregated across the document:

```text
[positive_document_score,
 negative_document_score]
```

This two-dimensional sentiment representation serves as input to both neural-network models.

---

# Classification Neural Network

## Objective

The classification model learns sentiment scores that maximize performance on a binary target variable.

The resulting dictionary is saved as:

```text
dictionary_classificationNN.pckl
```
## Architecture

```text
Document Sentiment Vector
            ↓
      Linear Layer (D)
            ↓
          tanh
            ↓
      Linear Layer (W)
            ↓
         Softmax
            ↓
   Binary Class Probabilities
```
During training both network parameters and dictionary sentiment values are updated through backpropagation.

---

# Regression Neural Network

## Objective

The regression model learns sentiment scores that explain a continuous financial target.

The resulting dictionary is saved as:

```text
dictionary_regressionNN.pckl
```

## Architecture

```text
Document Sentiment Vector
            ↓
      Linear Layer (D)
            ↓
          tanh
            ↓
   Positive Component
      minus
   Negative Component
            ↓
      Continuous Output
```

As with the classification model, word-level sentiment values are updated directly during optimization.

---

# Dictionary Learning

A key feature of Fin-SentiLex is that the sentiment dictionary itself is trainable.

Rather than learning only model coefficients, the neural networks continuously update:

```python
dictionary[word]['pos']
dictionary[word]['neg']
```

through gradient-based optimization.

Words associated with positive outcomes receive larger positive sentiment values, while words associated with negative outcomes receive larger negative sentiment values.

The final output is a finance-specific sentiment lexicon learned directly from historical filing data.

---

# SVM Evaluation

The generated dictionaries are benchmarked against the Loughran-McDonald dictionary using Support Vector Machines and Support Vector Regression.

The following dictionaries are evaluated:

```text
Loughran-McDonald
Benchmark NN
Classification NN
Regression NN
```

## Train-Test Split

A chronological split is used:

| Period | Usage |
|----------|--------|
| 2000-2014 | Training |
| 2015-2018 | Testing |

## Classification Tasks

Binary targets are evaluated using SVM models with a sigmoid kernel.

Evaluation metrics:

- Precision
- Recall
- F1 Score
- Accuracy

## Regression Tasks

Continuous targets are evaluated using NuSVR models with an RBF kernel.

Evaluation metric:

- RMSE

---

# Running the Pipeline

This document explains how to train new sentiment dictionaries and evaluate them using the Fin-SentiLex framework.

---

# Overview

The framework consists of two stages:

1. Dictionary construction.
2. Dictionary evaluation.

The complete workflow is:

```text
Initial Dictionary
       ↓
Classification NN
       ↓
Classification Dictionary
       ↓
Regression NN
       ↓
Regression Dictionary
       ↓
SVM/SVR Evaluation
       ↓
Results
```

---

# Required Files

Before running the pipeline, the following files should be available in the data directory:

```text
dictionary_filtered.pckl
Loughran_McDonald_dict.pckl

200010X_final.pckl
...
201810X_final.pckl
```

The location of these files is controlled by:

```python
drive = "/Volumes/LaCie/Data/"
```

Update this path if necessary.

---

# Step 1: Train the Classification Dictionary

The classification network learns word sentiment values that maximize performance on a binary target variable.

## Script

```bash
python classificationNN.py
```

## Training Data

```python
for year in range(2007, 2015)
```

Training period:

```text
2007-2014
```

## Main Parameters

```python
batch_size = 40
epochs = 1
```

| Parameter | Description | Value |
|------------|------------|--------|
| Batch size | Documents per update | 40 |
| Epochs | Passes through yearly dataset | 1 |
| Optimizer | Parameter updates | Adam |
| Loss | Objective function | Cross-Entropy |
| Activation | Hidden layer activation | tanh |

## Output Files

```text
dictionary_classificationNN.pckl
coefficients_classificationNN.pckl
Ms_classificationNN.pckl
Vs_classificationNN.pckl
```

---

# Step 2: Train the Regression Dictionary

The regression network learns sentiment values that explain a continuous target variable.

## Script

```bash
python regressionNN.py
```

## Training Data

```python
for year in range(2013, 2015)
```

Training period:

```text
2013-2014
```

## Main Parameters

```python
batch_size = 40
epochs = 1
```

| Parameter | Description | Value |
|------------|------------|--------|
| Batch size | Documents per update | 40 |
| Epochs | Passes through yearly dataset | 1 |
| Optimizer | Parameter updates | Adam |
| Loss | Objective function | MSE |
| Activation | Hidden layer activation | tanh |

## Output Files

```text
dictionary_regressionNN.pckl
```

---

# Step 3: Build the Evaluation Dictionaries

The SVM evaluation compares several dictionaries:

```python
dict_names = [
    'Loughran',
    'Benchmark',
    'Classification',
    'Regression'
]
```

The dictionaries are loaded and filtered through:

```python
svmDictionaries()
```

which loads:

```text
Loughran_McDonald_dict.pckl
dictionary_benchNN.pckl
dictionary_classificationNN.pckl
dictionary_regressionNN.pckl
```

and filters them using:

```python
fSVM.filterDicts(
    loughranDict,
    dictionaries,
    0.4
)
```

The value:

```python
0.4
```

is the current dictionary filtering threshold.

---

# Step 4: Build the Train/Test Dataset

Create the SVM dataset:

```python
train, test = SVMDataset(
    dictionaries,
    dict_names
)
```

## Training Period

```text
2000-2014
```

## Test Period

```text
2015-2018
```

The generated datasets can be stored as:

```python
fd.saveFile(train, drive+'train_final.pckl')
fd.saveFile(test, drive+'test_final.pckl')
```

---

# Step 5: Run Dictionary Evaluation

The main evaluation function is:

```python
forecastSVM(
    train,
    test,
    dict_names,
    'rbf',
    [0,1,2,3]
)
```

---

# Evaluation Targets

The framework evaluates four prediction tasks.

| Target | Type |
|----------|------|
| y = 0 | Classification |
| y = 1 | Regression |
| y = 2 | Regression |
| y = 3 | Classification |

---

# Classification Evaluation

For:

```python
y = 0
y = 3
```

the framework trains:

```python
runSVM(...)
```

using:

```text
Kernel = sigmoid
Parameter = 100
Parameter = 0.01
```

### Reported Metrics

```text
Precision (Positive)
Precision (Negative)

Recall (Positive)
Recall (Negative)

F1 (Positive)
F1 (Negative)

Accuracy
```

---

# Regression Evaluation

For:

```python
y = 1
y = 2
```

the framework trains:

```python
NuSVR(
    kernel='rbf',
    gamma='auto'
)
```

### Reported Metric

```text
RMSE
```

---

# Generating the Results Tables

Run:

```python
results = resultsForecasts(
    forecasts,
    dict_names,
    [0,1,2,3]
)
```

This returns a results table for each target.

For classification tasks:

```text
Precision
Recall
F1
Accuracy
```

For regression tasks:

```text
RMSE
```

---

# Testing a New Dictionary

Suppose a new dictionary has been trained and saved as:

```text
dictionary_my_model.pckl
```

The simplest way to test it is:

## 1. Load the dictionary

```python
myDict = fd.loadFile(
    drive+'dictionary_my_model.pckl'
)
```

## 2. Add it to the evaluation set

```python
dictionaries = [
    loughranDict,
    benchNNDict,
    classNNDict,
    regresNNDict,
    myDict
]

dict_names = [
    'Loughran',
    'Benchmark',
    'Classification',
    'Regression',
    'MyModel'
]
```

## 3. Create train/test scores

```python
train,test = SVMDataset(
    dictionaries,
    dict_names
)
```

## 4. Generate forecasts

```python
forecasts = forecastSVM(
    train,
    test,
    dict_names,
    'rbf',
    [0,1,2,3]
)
```

## 5. Generate performance tables

```python
results = resultsForecasts(
    forecasts,
    dict_names,
    [0,1,2,3]
)
```

The resulting tables can be directly compared against the benchmark dictionaries.

---

# Hyperparameter Summary

## Classification NN

| Parameter | Value |
|------------|---------|
| Training years | 2007-2014 |
| Batch size | 40 |
| Epochs | 1 |
| Loss | Cross-Entropy |
| Optimizer | Adam |
| Activation | tanh |

## Regression NN

| Parameter | Value |
|------------|---------|
| Training years | 2013-2014 |
| Batch size | 40 |
| Epochs | 1 |
| Loss | MSE |
| Optimizer | Adam |
| Activation | tanh |

## SVM

| Parameter | Value |
|------------|---------|
| Train period | 2000-2014 |
| Test period | 2015-2018 |
| Classification kernel | Sigmoid |
| SVR kernel | RBF |
| SVR gamma | Auto |
