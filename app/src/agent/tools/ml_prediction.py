
"""ML Prediction Tool for CreditGuard AI."""

import pandas as pd

from src.model_inference import predict_fraud


def run_ml_prediction(
    transaction: pd.DataFrame,
    threshold: float = 0.50,
) -> dict:
    """
    Generate fraud prediction using the trained XGBoost S1 model.

    Input:
        transaction: Raw transaction data containing 30 features.
        threshold: Classification threshold.

    Output:
        Structured prediction result.
    """

    if not isinstance(transaction, pd.DataFrame):
        raise TypeError("Transaction must be a pandas DataFrame.")

    if len(transaction) != 1:
        raise ValueError(
            "ML Prediction Tool currently processes one transaction at a time."
        )

    result = predict_fraud(
        transaction=transaction,
        threshold=threshold,
        already_scaled=False,
    )

    probability = result["probabilities"][0]
    prediction = result["predictions"][0]

    return {
        "tool": "ml_prediction",
        "status": "success",
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "threshold": float(threshold),
        "classification": (
            "FRAUD" if prediction == 1 else "NORMAL"
        ),
    }
