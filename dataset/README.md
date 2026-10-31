<p align="center">
  <img src="https://img.shields.io/badge/DATASET-CREDIT%20CARD%20FRAUD-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Source-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
</p>

<h1 align="center">Dataset</h1>

<p align="center">
  Dataset management for Credit Card Fraud Detection research.
</p>

---

## About Dataset

This directory contains the datasets used in the credit card fraud detection research. The dataset is obtained from the [European Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset on Kaggle.

## Dataset Structure

```text
dataset/
│
├── preprocessed/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_val.csv
│   ├── y_val.csv
│   ├── X_test.csv
│   └── y_test.csv
│
└── balanced/
    ├── S2/
    └── S3/
```

## Directory Description

| Directory | Description |
|---|---|
| `raw/` | Original dataset obtained from Kaggle. |
| `preprocessed/` | Dataset after preprocessing and train-validation-test splitting. |
| `balanced/S2/` | Balanced training data for Scenario 2. |
| `balanced/S3/` | Balanced training data for Scenario 3. |

## Dataset Information

| Attribute | Description |
|---|---|
| Total Transactions | 284,807 |
| Input Features | 30 |
| Target Variable | `Class` |
| Class 0 | Non-Fraud |
| Class 1 | Fraud |

## Data Source

**European Credit Card Fraud Detection**

Source: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Note:** The original dataset (`creditcard.csv`) is not included in the GitHub repository due to its large file size. It can be downloaded directly from the original Kaggle source.
