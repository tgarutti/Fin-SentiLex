# Fin-SentiLex

Implementation for the automatic construction and evaluation of finance-specific sentiment lexicons using SEC 10-X filings.

Fin-SentiLex builds sentiment dictionaries from financial disclosures and evaluates them using neural-network based dictionary learning and Support Vector Machine (SVM) benchmarking. The primary objective is to compare manually curated financial lexicons with dictionaries learned directly from historical filing data.

---

## Overview

The framework consists of three main stages:

1. Text processing and feature extraction from SEC filings.
2. Dictionary construction using custom neural-network models.
3. Dictionary evaluation using SVM and SVR models.

The workflow is illustrated below:

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

## Repository Structure

```text
Fin-SentiLex/
│
├── dictionaries/
│
├── docs/
│   ├── methodology.md
│   ├── running_the_pipeline.md
│   └── dictionary_format.md
│
├── get10Xlist.py
├── getFileList10X.py
├── searchFiles.py
│
├── parsingData.py
├── functionsData.py
├── functions10X.py
│
├── classificationNN.py
├── regressionNN.py
├── functionsNN.py
│
├── SVMs.py
├── functionsSVM.py
│
├── benchmarkDict.py
├── benchmarkNN.py
│
└── README.md
```

---

## Quick Start

### Train the Classification Dictionary

```bash
python classificationNN.py
```

Output:

```text
dictionary_classificationNN.pckl
```

### Train the Regression Dictionary

```bash
python regressionNN.py
```

Output:

```text
dictionary_regressionNN.pckl
```

### Build the Evaluation Dataset

```python
train, test = SVMDataset(
    dictionaries,
    dict_names
)
```

### Run Dictionary Evaluation

```python
forecasts = forecastSVM(
    train,
    test,
    dict_names,
    'rbf',
    [0,1,2,3]
)
```

### Generate Performance Tables

```python
results = resultsForecasts(
    forecasts,
    dict_names,
    [0,1,2,3]
)
```

---

## Dictionaries Evaluated

The framework compares four dictionary construction approaches:

| Dictionary | Description |
|------------|-------------|
| Loughran-McDonald | Standard manually curated finance sentiment lexicon |
| Benchmark NN | Baseline neural-network dictionary |
| Classification NN | Dictionary optimized for classification tasks |
| Regression NN | Dictionary optimized for regression tasks |

---

## Outputs

The pipeline produces:

```text
dictionary_classificationNN.pckl
dictionary_regressionNN.pckl

train_final.pckl
test_final.pckl

forecasts.pckl
```

along with summary evaluation tables containing:

- Precision
- Recall
- F1 Score
- Accuracy
- RMSE

---

## Documentation

Detailed documentation is available in the `docs/` directory.

### Methodology

Description of the neural-network architectures, dictionary learning framework, and benchmarking methodology.

```text
docs/methodology.md
```

### Running the Pipeline

Step-by-step instructions for training dictionaries and reproducing the benchmark results.

```text
docs/running_the_pipeline.md
```

### Dictionary and Data Formats

Description of dictionary structures, dataset formats, and benchmark inputs.

```text
docs/dictionary_format.md
```

---

## Requirements

The project was developed in Python using standard scientific-computing libraries.

Main dependencies:

```bash
pip install numpy
pip install pandas
pip install scikit-learn
```

Additional dependencies may be required depending on the data collection and text-processing workflow.

---
