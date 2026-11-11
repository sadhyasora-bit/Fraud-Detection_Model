
import joblib
from pathlib import Path

scaler_path = Path("models/scaler_s1_s2.joblib")
scaler = joblib.load(scaler_path)

print("Scaler berhasil dimuat.")
print("Scaler type:", type(scaler).__name__)
print("Jumlah fitur:", scaler.n_features_in_)
print("Feature names:", scaler.feature_names_in_)

