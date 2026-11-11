
import joblib
import pandas as pd

from src.ml_data import load_scenario_splits
from src.utils import DATA_DIR
from src.agent.tools.ml_prediction import run_ml_prediction


# 1. Load S1 validation data
splits = load_scenario_splits(
    DATA_DIR / "preprocessed",
    "S1",
    include_test=False,
)

# 2. Find one Normal and one Fraud sample
normal_index = splits.y_val[splits.y_val == 0].index[0]
fraud_index = splits.y_val[splits.y_val == 1].index[0]

samples = {
    "NORMAL": splits.X_val.loc[[normal_index]],
    "FRAUD": splits.X_val.loc[[fraud_index]],
}

# 3. Load S1 scaler
scaler = joblib.load("models/scaler_s1_s2.joblib")

# 4. Test both samples
for actual_class, scaled_sample in samples.items():

    # Reconstruct raw transaction
    raw_values = scaler.inverse_transform(scaled_sample)

    raw_transaction = pd.DataFrame(
        raw_values,
        columns=scaled_sample.columns,
    )

    # Run ML Prediction Tool
    result = run_ml_prediction(raw_transaction)

    # Display result
    print(f"\n===== ACTUAL CLASS: {actual_class} =====")

    for key, value in result.items():
        print(f"{key}: {value}")
