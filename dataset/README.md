 <p align="center">
  <img src="https://img.shields.io/badge/DATASET-CREDIT%20CARD%20FRAUD-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Source-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/Storage-Google%20Drive-34A853?style=for-the-badge&logo=googledrive&logoColor=white" />
</p>

<h1 align="center">Dataset</h1>

<p align="center">
  Dataset management for Credit Card Fraud Detection research.
</p>

---

## About Dataset

This directory documents the datasets used in the Credit Card Fraud Detection research. The original dataset is obtained from the [European Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset on Kaggle.

Due to the large file sizes, the original, preprocessed, and scenario-specific datasets are stored externally in Google Drive rather than directly in this GitHub repository.

## Dataset Structure

The following structure represents the dataset organization in Google Drive:

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
└── scenarios/
    │
    ├── S1_Original/
    │   ├── X_train.csv
    │   ├── y_train.csv
    │   ├── X_val.csv
    │   ├── y_val.csv
    │   ├── X_test.csv
    │   └── y_test.csv
    │
    ├── S2_Augmentation/
    │   ├── X_train.csv
    │   ├── y_train.csv
    │   ├── X_val.csv
    │   ├── y_val.csv
    │   ├── X_test.csv
    │   └── y_test.csv
    │
    └── S3_Preprocessing_Augmentation/
        ├── X_train.csv
        ├── y_train.csv
        ├── X_val.csv
        ├── y_val.csv
        ├── X_test.csv
        └── y_test.csv
```

## Directory Description

| Directory | Description |
|---|---|
| `preprocessed/` | Dataset after initial preprocessing and train-validation-test splitting. |
| `scenarios/S1_Original/` | Original preprocessed dataset without additional augmentation. |
| `scenarios/S2_Augmentation/` | Dataset prepared using the augmentation approach. |
| `scenarios/S3_Preprocessing_Augmentation/` | Dataset prepared using the combined preprocessing and augmentation approach. |

Each scenario contains separate feature (`X`) and target (`y`) files for training, validation, and testing.

## Dataset Information

| Attribute | Description |
|---|---|
| Total Transactions | 284,807 |
| Input Features | 30 |
| Target Variable | `Class` |
| Class 0 | Non-Fraud |
| Class 1 | Fraud |

## Data Handling

- Features (`X`) and target labels (`y`) are stored separately.
- Dataset splitting uses stratification to preserve class proportions.
- Data augmentation or resampling is applied only to training data to prevent data leakage.
- Validation and testing data retain their original class distribution for model evaluation.

## Data Source

**European Credit Card Fraud Detection**

Source: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Dataset Access

<p align="center">
  <a href="https://drive.google.com/drive/folders/1yynwraXJnBuL88eKJywPiwpXzJkP4lLt?usp=sharing">
    <img src="https://img.shields.io/badge/Google%20Drive-Access%20Dataset-34A853?style=for-the-badge&logo=googledrive&logoColor=white" />
  </a>
</p>

**Note:** Dataset files are not included in this GitHub repository due to their large storage requirements. Access the Google Drive folder above to download the original, preprocessed, and scenario-specific datasets.
