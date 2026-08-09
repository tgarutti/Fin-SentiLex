# Methodology

## Overview

Fin-SentiLex is a framework for the automatic construction and evaluation of finance-specific sentiment lexicons using SEC 10-X filings.

The framework consists of three main stages:

1. Text processing.
2. Dictionary construction.
3. Dictionary evaluation.

The overall workflow is illustrated below.

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

# Text Processing

SEC filings are cleaned and tokenized using the utilities provided in `functions10X.py`.

For each document:

1. Raw text is cleaned and normalized.
2. Tokens are extracted.
3. Dictionary words are identified.
4. TF-IDF weighting is applied.
5. Positive and negative sentiment scores are aggregated.

Each document is therefore represented by a low-dimensional sentiment representation derived from its underlying word composition.

---

# Sentiment Dictionary

Every word in the dictionary is assigned positive and negative sentiment values:

```python
dictionary[word]["pos"]
dictionary[word]["neg"]
```

These values are initialized from a starting dictionary and subsequently updated during neural-network training.

Unlike traditional sentiment-analysis approaches, dictionary values are treated as trainable parameters.

---

# Document Representation

For each document, dictionary words are converted into sentiment vectors:

```text
[word_positive_score,
 word_negative_score]
```

After TF-IDF weighting, the word vectors are aggregated to obtain:

```text
[positive_document_score,
 negative_document_score]
```

This two-dimensional representation serves as the input to both neural-network architectures.

---

# Classification Neural Network

## Objective

The classification model learns sentiment scores that maximize predictive performance on a binary target variable.

The resulting dictionary is stored as:

```text
dictionary_classificationNN.pckl
```

## Architecture

```text
Document Sentiment Vector
            ↓
      Linear Layer
            ↓
          tanh
            ↓
      Linear Layer
            ↓
         Softmax
            ↓
   Binary Class Probabilities
```

The model is optimized using cross-entropy loss and Adam updates.

Both model parameters and word-level sentiment scores are updated during training.

---

# Regression Neural Network

## Objective

The regression model learns sentiment scores that explain a continuous financial target.

The resulting dictionary is stored as:

```text
dictionary_regressionNN.pckl
```

## Architecture

```text
Document Sentiment Vector
            ↓
      Linear Layer
            ↓
          tanh
            ↓
 Positive Signal
      minus
 Negative Signal
            ↓
    Continuous Output
```

The model is optimized using mean squared error and Adam updates.

As with the classification model, sentiment values within the dictionary are updated directly during training.

---

# Dictionary Learning

A key feature of Fin-SentiLex is that the dictionary itself is learned from data.

During training, the sentiment values assigned to each word are updated through gradient-based optimization.

Words associated with positive outcomes tend to receive stronger positive sentiment values, while words associated with negative outcomes tend to receive stronger negative sentiment values.

The final output is therefore a finance-specific sentiment lexicon learned directly from historical filing data.

---

# Dictionary Evaluation

The generated dictionaries are benchmarked against alternative sentiment lexicons using Support Vector Machines (SVM) and Support Vector Regression (SVR).

The framework compares:

```text
Loughran-McDonald
Benchmark NN
Classification NN
Regression NN
```

A chronological train-test split is used:

| Period | Usage |
|----------|--------|
| 2000-2014 | Training |
| 2015-2018 | Testing |

---

# Evaluation Metrics

## Classification Tasks

Classification performance is evaluated using:

- Precision
- Recall
- F1 Score
- Accuracy

## Regression Tasks

Regression performance is evaluated using:

- RMSE

The resulting metrics are used to compare the predictive performance of the different dictionary construction methods.
