"""Settings & Dataset — choose a local CSV and define its schema mapping."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ml_data import scenario_package_available
from src.utils import (
    DATA_DIR,
    PROJECT_ROOT,
    available_datasets,
    infer_positive_class,
    load_dataset,
    load_dataset_settings,
    map_target_labels,
    preview_csv,
    resolve_column_mapping,
    resolve_dataset_path,
    save_dataset_settings,
    target_values,
)


st.title("Settings & Dataset")
st.caption("Choose the active transaction dataset and define how its columns are interpreted.")

settings = load_dataset_settings()
current_path = resolve_dataset_path(settings)
existing_files = available_datasets()

with st.container(border=True):
    st.subheader("Add a dataset")
    st.caption("A saved CSV stays in the project data folder and loads automatically on future runs.")
    new_upload = st.file_uploader("Choose a CSV file to save locally", type=["csv"], key="settings_dataset_upload")
    if new_upload is not None and st.button("Save CSV to project", type="primary"):
        original_name = Path(new_upload.name).name
        stem = Path(original_name).stem or "transactions"
        destination = DATA_DIR / f"{stem}.csv"
        version = 1
        while destination.exists():
            destination = DATA_DIR / f"{stem}_{version}.csv"
            version += 1
        try:
            destination.write_bytes(new_upload.getvalue())
            st.success(f"Saved {destination.name} in the project data folder.")
            st.rerun()
        except OSError as exc:
            st.error(f"Could not save the dataset: {exc}")

existing_files = available_datasets()
if not existing_files:
    st.warning("No CSV file was found in the project data folder. Add one above or place it in data/.")
    st.stop()

selected_default = current_path if current_path in existing_files else existing_files[0]
selected_path = st.selectbox(
    "Active dataset",
    options=existing_files,
    index=existing_files.index(selected_default),
    format_func=lambda path: path.name,
    key="settings_active_dataset",
)

if selected_path != current_path:
    st.info("This dataset will become active after you save the settings below.")

try:
    preview = preview_csv(selected_path, rows=12)
    dataset = load_dataset(selected_path)
except Exception as exc:
    st.error(f"Could not read {selected_path.name}: {exc}")
    st.stop()

columns = [str(column) for column in preview.columns]
using_saved_mapping = selected_path == current_path
mapping_settings = settings if using_saved_mapping else {}
suggested = resolve_column_mapping(columns, mapping_settings)
st.subheader("Column mapping")
st.caption("Map the fields used by the dashboard. Transaction ID is optional; when absent, row numbers are used.")

map_col_a, map_col_b = st.columns(2)
with map_col_a:
    transaction_id = st.selectbox(
        "Transaction ID",
        options=[None, *columns],
        index=([None, *columns].index(suggested["transaction_id"])),
        format_func=lambda value: "Not mapped" if value is None else value,
        key=f"mapping_id_{selected_path.name}",
    )
    time_column = st.selectbox(
        "Time",
        options=[None, *columns],
        index=([None, *columns].index(suggested["time"])),
        format_func=lambda value: "Not mapped" if value is None else value,
        key=f"mapping_time_{selected_path.name}",
    )
with map_col_b:
    amount_column = st.selectbox(
        "Amount",
        options=[None, *columns],
        index=([None, *columns].index(suggested["amount"])),
        format_func=lambda value: "Not mapped" if value is None else value,
        key=f"mapping_amount_{selected_path.name}",
    )
    target_column = st.selectbox(
        "Actual label / target",
        options=[None, *columns],
        index=([None, *columns].index(suggested["target"])),
        format_func=lambda value: "Not mapped" if value is None else value,
        key=f"mapping_target_{selected_path.name}",
    )

positive_class: str | None = None
if target_column:
    values = target_values(dataset[target_column])
    if values:
        saved_positive = settings.get("positive_class") if using_saved_mapping else None
        suggested_positive = saved_positive if saved_positive in values else infer_positive_class(values)
        default_index = values.index(suggested_positive) if suggested_positive in values else 0
        positive_class = st.selectbox(
            "Positive class (fraud)",
            options=values,
            index=default_index,
            help="Rows with this actual target value are counted as Fraud; other non-empty values are counted as Normal.",
            key=f"positive_class_{selected_path.name}_{target_column}",
        )
    else:
        st.warning("The selected target column has no non-empty values.")

mapping = {
    "transaction_id": transaction_id,
    "time": time_column,
    "amount": amount_column,
    "target": target_column,
}
if st.button("Save dataset settings", type="primary"):
    try:
        settings_to_save = {
            "dataset_path": selected_path.relative_to(PROJECT_ROOT).as_posix(),
            "column_mapping": mapping,
            "positive_class": positive_class,
        }
        save_dataset_settings(settings_to_save)
        st.success("Dataset and column settings saved. The Dashboard will use these settings.")
        current_path = selected_path
        settings = settings_to_save
        using_saved_mapping = True
    except OSError as exc:
        st.error(f"Could not save dataset settings: {exc}")

st.divider()
st.subheader("Dataset validation")
missing_values = int(dataset.isna().sum().sum())
duplicate_rows = int(dataset.duplicated().sum())
stats = st.columns(4)
stats[0].metric("Rows", f"{len(dataset):,}")
stats[1].metric("Columns", f"{len(dataset.columns):,}")
stats[2].metric("Missing values", f"{missing_values:,}")
stats[3].metric("Duplicate rows", f"{duplicate_rows:,}")

validation_messages: list[tuple[str, bool]] = [
    ("Dataset contains rows", len(dataset) > 0),
    ("Amount column is mapped", amount_column is not None),
    ("Actual label column is mapped", target_column is not None),
]
for message, okay in validation_messages:
    (st.success if okay else st.warning)(message)

if target_column and positive_class is not None:
    actual_labels = map_target_labels(dataset[target_column], positive_class)
    class_counts = actual_labels.value_counts().rename_axis("Actual Label").reset_index(name="Transactions")
    class_counts["Share"] = (class_counts["Transactions"] / len(dataset) * 100).map(lambda value: f"{value:.3f}%")
    st.markdown("**Actual label distribution**")
    st.dataframe(class_counts, hide_index=True, width="stretch", height=160, alt="Actual labels in the active dataset")

with st.container(border=True):
    st.subheader("ML training preparation")
    st.caption(
        "CreditGuard will reuse the existing fraud-preprocessing notebook outputs. "
        "The raw CSV and Dashboard counts stay unchanged."
    )
    if scenario_package_available(DATA_DIR / "preprocessed"):
        st.success("Prepared scenario split files found in data/preprocessed. They can be loaded without rerunning cleaning, splitting, scaling, or resampling.")
    else:
        st.warning("Prepared scenario split CSVs are not in data/preprocessed yet. Copy the existing notebook export there; CreditGuard will not recreate the split or preprocessing.")
    st.markdown(
        "**Existing recipe:** remove exact duplicates in the derived research data; stratified random split 70/15/15 "
        "(seed 42); RobustScaler fitted on train only. S1 has no resampling; S2 applies SMOTE to train only; "
        "S3 applies log1p to Amount, then RobustScaler and training-only SMOTE. Validation and test remain untouched."
    )
    st.caption("See data/preprocessed/README.txt for the expected folder layout. Model predictions and evaluation metrics have not been added here.")

with st.container(border=True):
    st.subheader("Dataset preview")
    preview_columns = list(dict.fromkeys(
        column for column in (transaction_id, time_column, amount_column, target_column) if column
    ))
    if not preview_columns:
        preview_columns = columns[:8]
    st.dataframe(
        preview.loc[:, preview_columns],
        hide_index=True,
        width="stretch",
        height=320,
        alt="Preview of the first rows in the active dataset",
    )


