
import pandas as pd

from src.model_inference import load_fraud_model, predict_fraud
from src.ml_data import load_scenario_splits
from src.utils import DATA_DIR


# Load the trained model
model = load_fraud_model()
print("Model loaded successfully.")

# Load one validation sample from S1
splits = load_scenario_splits(
    DATA_DIR / "preprocessed",
    "S1",
    include_test=False
)

sample = splits.X_val.iloc[[0]]

# Run prediction
result = predict_fraud(sample)

print("\n===== INFERENCE RESULT =====")
print("Probability:", result["probabilities"][0])
print("Prediction:", result["predictions"][0])
print("Threshold:", result["threshold"])
print("Total transactions:", result["total_transactions"])
