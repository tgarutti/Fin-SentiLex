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

## Training Period

The model is trained on filings from:

```text
2007-2014
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

## Training

| Parameter | Value |
|------------|------------|
| Batch size | 40 |
| Epochs | 1 |
| Optimizer | Adam |
| Activation | tanh |
| Loss | Cross-Entropy |

During training both network parameters and dictionary sentiment values are updated through backpropagation.

---

# Regression Neural Network

## Objective

The regression model learns sentiment scores that explain a continuous financial target.

The resulting dictionary is saved as:

```text
dictionary_regressionNN.pckl
```

## Training Period

The model is trained on filings from:

```text
2013-2014
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

## Training

| Parameter | Value |
|------------|------------|
| Batch size | 40 |
| Epochs | 1 |
| Optimizer | Adam |
| Activation | tanh |
| Loss | Mean Squared Error |

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

# Experimental Objective

The project evaluates whether automatically learned sentiment lexicons provide better predictive performance than traditional manually curated financial dictionaries.

In particular, the framework investigates:

1. Whether sentiment scores can be learned directly from financial outcomes.
2. Whether learned dictionaries outperform benchmark lexicons.
3. Whether dictionaries optimized for classification differ from those optimized for regression.

The resulting performance differences provide empirical evidence on the value of automated financial sentiment dictionary construction.
