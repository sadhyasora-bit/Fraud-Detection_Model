 <p align="center">
  <img src="https://img.shields.io/badge/Dataset-Credit%20Card%20Fraud-blue?style=for-the-badge" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Source-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle"/>
</p>

<h1 align="center">Credit Card Fraud Detection Dataset</h1>

<p align="center">
  Dataset used for credit card fraud detection research.
</p>

---

## About Dataset

This directory contains the dataset used in the Credit Card Fraud Detection research.

The original dataset is obtained from Kaggle:

**Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Dataset Information

| Description | Value |
|---|---|
| Total Transactions | 284,807 |
| Input Features | 30 |
| Target Variable | Class |
| Class 0 | Non-Fraud |
| Class 1 | Fraud |

## Directory Structure

```text
dataset/
├── README.md
└── split/
    ├── X_train.csv
    ├── X_val.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_val.csv
    └── y_test.csv
```

## Data Description

The `split/` directory contains the preprocessed dataset divided into three subsets:

- **Training:** Used for model training.
- **Validation:** Used for model validation and parameter selection.
- **Testing:** Used for final model evaluation.

Feature (`X`) and target (`y`) data are stored separately.

## Note

The original CSV dataset is not included in this repository due to its large file size. It can be downloaded directly from the Kaggle source provided above.
