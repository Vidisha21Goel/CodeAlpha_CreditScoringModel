# CodeAlpha_CreditScoringModel

**Task:** Predict an individual's creditworthiness using past financial data.

## Overview

This project builds and compares three classification models — **Logistic
Regression**, **Decision Tree**, and **Random Forest** — to predict whether
a person is a good credit risk (`creditworthy = 1`) or not (`0`), based on
financial-history features such as income, debt, payment history, credit
utilization, and savings.

## About the dataset

No dataset was provided with this task, so `generate_data.py` **synthesizes
a realistic credit-history dataset** (5,000 samples) with financial
features and a target label derived from a weighted combination of
standard underwriting risk factors (income, debt-to-income ratio, late
payments, credit utilization, credit history length, savings, etc.), plus
noise so the classes aren't trivially separable.

If you'd rather train on a **real** dataset instead (recommended if you
want to showcase this on your resume/LinkedIn), two good free options are:
- [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
- [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

Just download either one, rename/re-map its target column to
`creditworthy` (1 = good risk / 0 = bad risk), save it as
`data/credit_data.csv`, and skip `generate_data.py`.

## Project structure

```
CodeAlpha_CreditScoringModel/
├── generate_data.py     # creates the synthetic dataset (data/credit_data.csv)
├── train_model.py        # feature engineering, training, evaluation, plots
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

python generate_data.py     # generates data/credit_data.csv
python train_model.py       # trains all 3 models, evaluates, saves outputs/
```

## What `train_model.py` does

1. Loads the dataset and engineers a few extra features (income-to-loan
   ratio, savings-to-income ratio, late payments per credit year).
2. Splits into train/test (80/20, stratified).
3. Scales features for Logistic Regression.
4. Trains **Logistic Regression**, **Decision Tree**, and **Random Forest**.
5. Evaluates each with **Precision, Recall, F1-Score, ROC-AUC, Accuracy**.
6. Saves confusion matrices, ROC curves, a feature-importance chart, and
   the trained models (`.pkl`) to `outputs/`.

## Sample results

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|----------|-----------|--------|------|---------|
| Logistic Regression | 0.80     | 0.81      | 0.79   | 0.80 | 0.87    |
| Decision Tree        | 0.73     | 0.73      | 0.74   | 0.73 | 0.79    |
| Random Forest        | 0.78     | 0.78      | 0.77   | 0.77 | 0.85    |

*(exact numbers will vary slightly by random seed / dataset version — see
`outputs/results.json` after running).*

## Key learnings demonstrated

- Feature engineering from raw financial history
- Handling class balance and stratified splitting
- Comparing linear vs. tree-based classifiers
- Standard classification evaluation metrics (Precision/Recall/F1/ROC-AUC)
- Model interpretability via feature importance
