# CreditGuard AI dataset contract

## Raw dashboard dataset

- File: `data/creditcard.csv`.
- 284,807 rows and 31 columns: `Time`, `V1`–`V28`, `Amount`, and `Class`.
- `Class=1` is the observed Fraud label (492 rows); `Class=0` is Normal (284,315 rows).
- The raw file has 0 missing values and 1,081 exact duplicate rows. It remains unchanged; dashboard counts describe this raw file.
- No transaction ID column exists. Dashboard row numbers are display references, not model features.

## Existing preprocessing from `fraud-preprocessing`

CreditGuard should consume the notebook's exported split files instead of repeating preparation. The notebook work uses:

1. Remove exact duplicate rows in the derived research dataset; preserve the raw CSV.
2. Stratified random train/validation/test split at 70%/15%/15%, with `random_state=42`.
3. Fit `RobustScaler` on train only for S1 and S2; apply it to validation and test.
4. S1: scaled original data, no resampling.
5. S2: apply SMOTE (`random_state=42`, `k_neighbors=5`) to training only.
6. S3: apply `log1p(Amount)`, fit a separate `RobustScaler` on train only, then SMOTE on training only.
7. Validation and test remain unresampled and preserve their original class distribution.

Expected split sizes from the notebook: train 198,608; validation 42,559; test 42,559. The validation and test sets each contain 42,488 Normal and 71 Fraud records. These counts are for the duplicate-filtered derived data; the dashboard continues to report raw source counts.

No model predictions or evaluation metrics are included in this preprocessing. CreditGuard must keep actual labels clearly distinct from future model predictions.

## Reuse package

The current project contains the existing split CSVs under `data/preprocessed/` with the three scenario folders directly inside it. `src/ml_data.py` accepts those folders directly or under a `scenarios/` subfolder. Each scenario requires X and y files for train, validation, and test. Metadata and scaler artifacts are optional when consuming these prepared CSVs because the feature values are already transformed. The loader only reads and checks the prepared files; it does not clean, split, scale, or resample them.

## Evaluation

When model work begins, use validation for model/threshold choices and keep test results untouched until final evaluation. Report the positive class explicitly and include confusion matrix, precision, recall, F1, specificity, PR-AUC, and ROC-AUC. Metrics have not been computed yet.
