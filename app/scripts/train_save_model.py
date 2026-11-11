
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.ml_data import load_scenario_splits
from src.utils import DATA_DIR, PROJECT_ROOT


def main():
    # Load S1 training and validation data only
    splits = load_scenario_splits(
        DATA_DIR / "preprocessed",
        "S1",
        include_test=False
    )

    X_fit = pd.concat(
        [splits.X_train, splits.X_val],
        ignore_index=True
    )

    y_fit = pd.concat(
        [splits.y_train, splits.y_val],
        ignore_index=True
    )

    positive_count = int((y_fit == 1).sum())
    negative_count = int((y_fit == 0).sum())

    # Final XGBoost configuration
    final_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=max(
            1.0,
            float(negative_count) / max(1, positive_count)
        ),
        random_state=42,
        n_jobs=-1,
    )

    print("Training final XGBoost S1...")
    final_model.fit(X_fit, y_fit)

    # Save trained model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_s1_final.json"

    if model_path.exists():
        raise FileExistsError(
            f"Model already exists: {model_path}"
        )

    final_model.save_model(str(model_path))

    print("Training completed.")
    print(f"Training rows: {len(X_fit):,}")
    print(f"Features: {len(X_fit.columns)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
