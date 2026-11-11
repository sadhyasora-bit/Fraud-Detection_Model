
"""Inference interface for the trained CreditGuard XGBoost S1 model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_s1_final.json"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler_s1_s2.joblib"

DEFAULT_THRESHOLD = 0.50


@st.cache_resource
def load_fraud_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    return model


@st.cache_resource
def load_s1_scaler():
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    return joblib.load(SCALER_PATH)


def predict_fraud(
    transaction: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    already_scaled: bool = False,
) -> dict:

    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not isinstance(transaction, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if transaction.empty:
        raise ValueError("Input cannot be empty.")

    model = load_fraud_model()
    expected_features = list(model.feature_names_in_)

    missing_features = [
        col for col in expected_features
        if col not in transaction.columns
    ]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    X = transaction.loc[:, expected_features].copy()

    if X.isna().any().any():
        raise ValueError("Input contains missing values.")

    if not already_scaled:
        scaler = load_s1_scaler()

        X = pd.DataFrame(
            scaler.transform(X),
            columns=expected_features,
            index=X.index,
        )

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype("int8")

    return {
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "threshold": threshold,
        "total_transactions": len(X),
    }
