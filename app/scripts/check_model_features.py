
from pathlib import Path
from xgboost import XGBClassifier

model_path = Path("models/xgboost_s1_final.json")

model = XGBClassifier()
model.load_model(str(model_path))

print("Model berhasil dimuat.")

print("\nFeature names dari model:")
print(model.feature_names_in_)

print("\nJumlah fitur:")
print(model.n_features_in_)
