
"""Transaction Analysis Tool for CreditGuard AI."""

import pandas as pd


def run_transaction_analysis(
    transaction: pd.DataFrame,
) -> dict:
    """
    Analyze factual characteristics of a single transaction.
    """

    if not isinstance(transaction, pd.DataFrame):
        raise TypeError(
            "Transaction must be a pandas DataFrame."
        )

    if len(transaction) != 1:
        raise ValueError(
            "Transaction Analysis Tool processes one transaction at a time."
        )

    required_features = ["Time", "Amount"]

    missing_features = [
        col for col in required_features
        if col not in transaction.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    time_value = float(transaction["Time"].iloc[0])
    amount_value = float(transaction["Amount"].iloc[0])

    return {
        "tool": "transaction_analysis",
        "status": "success",
        "transaction_amount": amount_value,
        "transaction_time": time_value,
        "amount_category": "SCALED_AMOUNT",
        "available_features": len(transaction.columns),
    }
