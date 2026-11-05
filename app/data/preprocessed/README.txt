CreditGuard AI prepared preprocessing files

Your existing layout is supported directly:

preprocessed/
  S1_Original/                 X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv
  S2_Augmentation/             same six files
  S3_Preprocessing_Augmentation/ same six files

The optional artifacts/ folder may contain scenario_metadata.json and the
scaler joblib files. It is not required to load these scenario CSVs: the X
files already contain prepared feature values and the y files contain the
corresponding labels.

CreditGuard's src/ml_data.py loads and validates these prepared files only.
It does not repeat duplicate removal, splitting, scaling, or SMOTE. The CSV
split files are sufficient to begin model training and evaluation using the
existing scenarios. Keep validation and test data untouched during training.
