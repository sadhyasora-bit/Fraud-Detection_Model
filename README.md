 <p align="center">
  <img src="https://img.shields.io/badge/PROJECT-CREDIT%20CARD%20FRAUD-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Agentic%20AI-Integration-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

<h1 align="center">Credit Card Fraud Detection</h1>

<p align="center">
  Machine Learning-Based Fraud Detection with Agentic AI Integration.
</p>

---

## About The Project

This project focuses on developing a credit card fraud detection system using machine learning algorithms and Agentic AI. The research investigates the performance of different classification algorithms under three experimental scenarios designed to evaluate data preprocessing and class imbalance handling strategies.

The study compares Logistic Regression, Random Forest, and XGBoost to identify the best-performing model for fraud detection. The selected model is then integrated with an Agentic AI approach to support fraud risk assessment and decision-making.

## Research Objectives

- Develop machine learning models for credit card fraud detection.
- Evaluate the impact of different data handling strategies on model performance.
- Compare Logistic Regression, Random Forest, and XGBoost across three experimental scenarios.
- Identify the best-performing model based on fraud detection evaluation metrics.
- Integrate the selected model with Agentic AI for fraud risk assessment and decision support.

## Dataset

The research uses the **European Credit Card Fraud Detection** dataset from Kaggle.

**Dataset Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Attribute | Description |
|---|---|
| Total Transactions | 284,807 |
| Input Features | 30 |
| Target Variable | `Class` |
| Class 0 | Non-Fraud |
| Class 1 | Fraud |

The original dataset, preprocessed data, and scenario-specific datasets are stored in Google Drive due to their large file sizes.

<p align="center">
  <a href="https://drive.google.com/drive/folders/1yynwraXJnBuL88eKJywPiwpXzJkP4lLt?usp=sharing">
    <img src="https://img.shields.io/badge/Google%20Drive-Access%20Dataset-34A853?style=for-the-badge&logo=googledrive&logoColor=white" />
  </a>
</p>

For dataset organization and access instructions, see [`dataset/README.md`](dataset/README.md).

## Experimental Scenarios

The research evaluates three experimental scenarios to investigate the effect of data handling strategies on classification performance.

| Scenario | Description |
|---|---|
| **S1 – Original** | Uses the original preprocessed dataset without additional augmentation. |
| **S2 – Augmentation** | Applies data augmentation to address class imbalance. |
| **S3 – Preprocessing + Augmentation** | Combines the specified preprocessing and augmentation procedures. |

Each scenario is evaluated using the same machine learning algorithms to enable performance comparison.

## Machine Learning Algorithms

Three classification algorithms are implemented:

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model used as a baseline. |
| Random Forest | Ensemble learning algorithm based on multiple decision trees. |
| XGBoost | Gradient boosting algorithm for classification tasks. |

## Model Evaluation

Model performance is assessed using multiple evaluation metrics, with particular attention to fraud detection and class imbalance.

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness. |
| Precision | Measures the proportion of predicted fraud cases that are actually fraud. |
| Recall (Sensitivity) | Measures the proportion of actual fraud cases correctly detected. |
| F1-Score | Harmonic mean of precision and recall. |
| Specificity | Measures the proportion of non-fraud cases correctly identified. |
| ROC-AUC | Measures the model's ability to distinguish between classes across classification thresholds. |
| PR-AUC | Evaluates the precision-recall trade-off, particularly relevant for imbalanced datasets. |
| Confusion Matrix | Summarizes true positive, true negative, false positive, and false negative predictions. |

## Agentic AI Integration

The best-performing machine learning model is integrated with an Agentic AI approach.

This stage extends the fraud detection process beyond classification by supporting fraud risk assessment and decision-making through an agent-based workflow.

The implementation and evaluation are documented in `07_Agentic_AI.ipynb`.

## Repository Structure

```text
Fraud-Detection_Model/
│
├── README.md
│
├── dataset/
│   └── README.md
│
├── artifacts/
│   ├── scaler_s1_s2.joblib
│   ├── scaler_s3.joblib
│   └── scenario_metadata.json
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Preprocessing_Analysis.ipynb
│   ├── 04_Imbalance_Handling.ipynb
│   ├── 05_Modeling.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_Agentic_AI.ipynb
│
└── results/
    ├── model_metrics/
    ├── confusion_matrices/
    └── agent_evaluation/
```

## Project Workflow

1. **Exploratory Data Analysis** – Explore dataset characteristics and class distribution.
2. **Data Preprocessing** – Prepare features, target labels, and dataset partitions.
3. **Preprocessing Analysis** – Analyze the characteristics of the preprocessed dataset.
4. **Imbalance Handling** – Prepare the three experimental scenarios.
5. **Modeling** – Train Logistic Regression, Random Forest, and XGBoost.
6. **Model Evaluation** – Compare model performance using classification metrics.
7. **Agentic AI** – Integrate and evaluate the selected machine learning model within an agent-based workflow.

## Technologies

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Imbalanced Learning:** Imbalanced-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook, Google Colab

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Fraud-Detection_Model.git
   ```

2. Download the dataset from the [Google Drive folder](https://drive.google.com/drive/folders/1yynwraXJnBuL88eKJywPiwpXzJkP4lLt?usp=sharing).

3. Place the downloaded dataset folders in the directory structure expected by the notebooks.

4. Install the required Python libraries:

   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib
   ```

5. Open the notebooks in numerical order and execute the workflow.

## Notes

- Dataset files are stored externally in Google Drive and are not included in this repository.
- Model artifacts are maintained separately in the `artifacts/` directory.
- Evaluation outputs are organized in the `results/` directory.
- Dataset access and directory details are available in [`dataset/README.md`](dataset/README.md).

---

<p align="center">
  <b>Credit Card Fraud Detection using Machine Learning and Agentic AI</b>
</p>
