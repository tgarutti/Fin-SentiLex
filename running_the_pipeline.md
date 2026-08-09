# Running the Pipeline

This document describes how to train sentiment dictionaries and evaluate them using the Fin-SentiLex framework.

---

# Overview

The framework consists of two stages:

1. Dictionary construction.
2. Dictionary evaluation.

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
SVM / SVR Evaluation
       ↓
Results
```

---

# Required Files

All scripts expect the data to be stored under:

```python
drive = "/Volumes/LaCie/Data/"
```

Update this path before running the code.

The following files are required:

```text
dictionary_filtered.pckl
Loughran_McDonald_dict.pckl

200010X_final.pckl
...
201810X_final.pckl
```

---

# Step 1 - Train the Classification Dictionary

Run:

```bash
python classificationNN.py
```

## Training Period

```text
2000-2014
```

## Main Parameters

```python
batch_size = 40
```

## Output

```text
dictionary_classificationNN.pckl
coefficients_classificationNN.pckl
Ms_classificationNN.pckl
Vs_classificationNN.pckl
```

---

# Step 2 - Train the Regression Dictionary

Run:

```bash
python regressionNN.py
```

## Training Period

```text
2000-2014
```

## Main Parameters

```python
batch_size = 40
```

## Output

```text
dictionary_regressionNN.pckl
```

---

# Step 3 - Prepare Dictionaries for Evaluation

The benchmark compares four dictionaries:

```python
dict_names = [
    "Loughran",
    "Benchmark",
    "Classification",
    "Regression"
]
```

The helper function:

```python
svmDictionaries()
```

loads:

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

---

# Step 4 - Build the Train/Test Dataset

Create the evaluation dataset:

```python
train, test = SVMDataset(
    dictionaries,
    dict_names
)
```

## Data Split

| Period | Usage |
|----------|----------|
| 2000-2014 | Training |
| 2015-2018 | Testing |

Optional:

```python
fd.saveFile(train, drive + "train_final.pckl")
fd.saveFile(test, drive + "test_final.pckl")
```

---

# Step 5 - Run Dictionary Evaluation

Run:

```python
forecasts = forecastSVM(
    train,
    test,
    dict_names,
    "rbf",
    [0,1,2,3]
)
```

---

# Prediction Targets

The framework evaluates four targets.

| Target | Model Type |
|----------|----------|
| y = 0 | Classification |
| y = 1 | Regression |
| y = 2 | Regression |
| y = 3 | Classification |

The exact definitions depend on the processed dataset.

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

Current configuration:

```text
Kernel = sigmoid
Parameter = 100
Parameter = 0.01
```

### Metrics

- Precision
- Recall
- F1 Score
- Accuracy

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
    kernel="rbf",
    gamma="auto"
)
```

### Metric

- RMSE

---

# Generate Results Tables

Create performance summaries:

```python
results = resultsForecasts(
    forecasts,
    dict_names,
    [0,1,2,3]
)
```

Classification results include:

```text
Precision
Recall
F1
Accuracy
```

Regression results include:

```text
RMSE
```

---

# Testing a New Dictionary

Suppose a newly trained dictionary exists:

```text
dictionary_my_model.pckl
```

## Load the Dictionary

```python
myDict = fd.loadFile(
    drive + "dictionary_my_model.pckl"
)
```

## Add It to the Benchmark

```python
dictionaries = [
    loughranDict,
    benchNNDict,
    classNNDict,
    regresNNDict,
    myDict
]

dict_names = [
    "Loughran",
    "Benchmark",
    "Classification",
    "Regression",
    "MyModel"
]
```

## Build Features

```python
train, test = SVMDataset(
    dictionaries,
    dict_names
)
```

## Run Forecasts

```python
forecasts = forecastSVM(
    train,
    test,
    dict_names,
    "rbf",
    [0,1,2,3]
)
```

## Generate Results

```python
results = resultsForecasts(
    forecasts,
    dict_names,
    [0,1,2,3]
)
```

The resulting tables can be compared directly against the benchmark dictionaries.

---

# Hyperparameter Summary

## Classification Neural Network

| Parameter | Value |
|------------|---------|
| Training Years | 2000-2014 |
| Batch Size | 40 |
| Optimizer | Adam |
| Loss | Cross-Entropy |
| Activation | tanh |

## Regression Neural Network

| Parameter | Value |
|------------|---------|
| Training Years | 2000-2014 |
| Batch Size | 40 |
| Optimizer | Adam |
| Loss | MSE |
| Activation | tanh |

## SVM Evaluation

| Parameter | Value |
|------------|---------|
| Training Period | 2000-2014 |
| Testing Period | 2015-2018 |
| Classification Kernel | Sigmoid |
| Regression Kernel | RBF |
| Regression Gamma | Auto |
