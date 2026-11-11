
import joblib
import numpy as np

from src.ml_data import load_scenario_splits
from src.model_inference import predict_fraud
from src.utils import DATA_DIR


# 1. Load S1 validation data (already scaled)
splits = load_scenario_splits(
    DATA_DIR / "preprocessed",
    "S1",
    include_test=False,
)

X_val = splits.X_val

# 2. Take one validation sample
scaled_sample = X_val.iloc[[0]]

# 3. Load S1 scaler
scaler = joblib.load("models/scaler_s1_s2.joblib")

# 4. Reconstruct raw sample
raw_values = scaler.inverse_transform(scaled_sample)

import pandas as pd

raw_sample = pd.DataFrame(
    raw_values,
    columns=scaled_sample.columns,
)

# 5. Predict using already-scaled data
result_scaled = predict_fraud(
    scaled_sample,
    already_scaled=True,
)

# 6. Predict using reconstructed raw data
result_raw = predict_fraud(
    raw_sample,
    already_scaled=False,
)

# 7. Verify scaling consistency
retransformed = scaler.transform(raw_sample)

scaling_match = np.allclose(
    retransformed,
    scaled_sample.to_numpy(),
    rtol=1e-5,
    atol=1e-5,
)

print("\n===== RAW INFERENCE TEST =====")
print("Scaling consistency:", scaling_match)

print("\nPrediction from scaled data:")
print("Probability:", result_scaled["probabilities"][0])
print("Prediction:", result_scaled["predictions"][0])

print("\nPrediction from raw data:")
print("Probability:", result_raw["probabilities"][0])
print("Prediction:", result_raw["predictions"][0])

print("\nPrediction consistency:",
      result_scaled["predictions"] == result_raw["predictions"])
