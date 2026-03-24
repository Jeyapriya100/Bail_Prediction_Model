# Indian Bail Judgment Prediction

Predicting bail outcomes from Indian court judgments using machine learning and NLP. This project explores two approaches — a classical SVM baseline and a fine-tuned legal BERT model — with a focus on **fairness and bias analysis**.

---

## Project Overview

Bail decisions in Indian courts involve complex legal reasoning. This project builds models to predict whether bail will be **Granted** or **Rejected**, and evaluates whether predictions differ based on the accused's gender (counterfactual fairness).

---

## Models

### 1. SVM Baseline (`svm.py`)
- Legal text cleaning (removes dates, IPC codes, citations)
- Lemmatization using spaCy
- TF-IDF with bigrams (8,000 features)
- One-hot encoded categorical features (court, region, gender, prior cases)
- `LinearSVC` classifier via sklearn Pipeline

### 2. Hybrid BERT Model (`bail_final_1.py`)
- Pretrained model: [`law-ai/InCaseLawBERT`](https://huggingface.co/law-ai/InCaseLawBERT)
- Combines BERT text embeddings with learned categorical embeddings
- Weighted cross-entropy loss for class imbalance
- Mixed precision training, linear warmup scheduler, early stopping
- Counterfactual gender fairness check
- Per-group accuracy and F1 evaluation

---

## Features Used

| Feature | Type |
|---|---|
| facts | Text |
| judgment_reason | Text |
| legal_issues | Text |
| court | Categorical |
| crime_type | Categorical |
| accused_gender | Categorical |
| region | Categorical |
| prior_cases | Categorical |

**Target:** `bail_outcome` (Granted / Rejected)

---

## Fairness Analysis

Both models are evaluated for gender bias using:
- **Counterfactual gender test** — same case, gender swapped Male ↔ Female, checking if prediction changes
- **Per-group metrics** — accuracy and macro-F1 reported separately for Male and Female groups
- **Case prioritization** — high-confidence and low-confidence (ambiguous) cases flagged for legal review

---

## Setup

```bash
pip install transformers datasets accelerate scikit-learn spacy torch
python -m spacy download en_core_web_sm
```

---

## Dataset

The dataset `indian_bail_judgments.csv` contains Indian court bail judgment records.

> **Note:** The dataset is not included in this repository due to privacy considerations. Place your CSV at the path specified in each script before running.

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM Baseline | 88 | 87 |
| InCaseLawBERT (Hybrid) | 95 | 94 |

*(Fill in after running evaluation)*

---

## Project Structure

```
├── bail_final_1.py      # Hybrid BERT model (InCaseLawBERT + categorical features)
├── svm.py               # SVM baseline (TF-IDF + LinearSVC)
└── README.md            # This file
```

---

## Acknowledgements

- [InCaseLawBERT](https://huggingface.co/law-ai/InCaseLawBERT) by law-ai for the domain-specific legal language model
- [Fairlearn](https://fairlearn.org/) for fairness evaluation concepts
