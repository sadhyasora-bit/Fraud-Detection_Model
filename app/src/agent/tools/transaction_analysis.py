
"""Transaction Analysis Tool for CreditGuard AI."""

import math
import pandas as pd


# Amount thresholds from original S1 training data (Notebook 02)
AMOUNT_Q1 = 5.70
AMOUNT_Q3 = 77.71
AMOUNT_P95 = 365.0365

# Time thresholds from original S1 training data (Notebook 02)
TIME_Q1 = 54273.75
TIME_MEDIAN = 84857.0
TIME_Q3 = 139372.25


def _categorize_amount(amount: float) -> str:
    """Categorize transaction amount using training distribution."""

    if amount <= AMOUNT_Q1:
        return "LOW"

    elif amount <= AMOUNT_Q3:
        return "MEDIUM"

    elif amount <= AMOUNT_P95:
        return "HIGH"

    return "VERY_HIGH"


def _categorize_time(time_value: float) -> str:
    """Categorize transaction time using training distribution."""

    if time_value <= TIME_Q1:
        return "EARLY"

    elif time_value <= TIME_MEDIAN:
        return "MID"

    elif time_value <= TIME_Q3:
        return "LATE"

    return "VERY_LATE"


def run_transaction_analysis(
    transaction: pd.DataFrame,
) -> dict:
    """
    Analyze factual characteristics of a single transaction.

    Input must use the original feature scale.
    """

    # 1. Validate input type
    if not isinstance(transaction, pd.DataFrame):
        raise TypeError(
            "Transaction must be a pandas DataFrame."
        )

    # 2. Validate transaction count
    if len(transaction) != 1:
        raise ValueError(
            "Transaction Analysis Tool processes one transaction at a time."
        )

    # 3. Validate required features
    required_features = ["Time", "Amount"]

    missing_features = [
        col for col in required_features
        if col not in transaction.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # 4. Extract and validate original values
    try:
        time_value = float(transaction["Time"].iloc[0])
        amount_value = float(transaction["Amount"].iloc[0])

    except (TypeError, ValueError):
        raise ValueError(
            "Time and Amount must contain numeric values."
        )

    if not math.isfinite(time_value) or not math.isfinite(amount_value):
        raise ValueError(
            "Time and Amount must contain finite values."
        )

    if time_value < 0:
        raise ValueError("Time cannot be negative.")

    if amount_value < 0:
        raise ValueError("Amount cannot be negative.")

    # 5. Categorize transaction characteristics
    amount_category = _categorize_amount(amount_value)
    time_category = _categorize_time(time_value)

    # 6. Generate descriptive indicators
    indicators = {
        "zero_amount": amount_value == 0,
        "high_amount": amount_category == "VERY_HIGH",
        "late_period": time_category == "VERY_LATE",
    }

    # 7. Return analysis result
    return {
        "tool": "transaction_analysis",
        "status": "success",
        "analysis_skipped": False,
        "reason": None,
        "transaction_amount": amount_value,
        "transaction_time": time_value,
        "amount_category": amount_category,
        "time_category": time_category,
        "transaction_indicators": indicators,
        "available_features": len(transaction.columns),
        "analysis_basis": "ORIGINAL_SCALE_TRAINING_DISTRIBUTION",
    }
