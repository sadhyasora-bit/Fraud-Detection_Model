"""CreditGuard AI — restored Dashboard Overview, styled from the supplied UI reference.

All figures are calculated from the uploaded dataset and its actual target labels.
This page does not create or display model predictions.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    infer_positive_class,
    load_dataset,
    load_dataset_settings,
    map_target_labels,
    read_uploaded_csv,
    resolve_column_mapping,
    resolve_dataset_path,
    target_values,
)


st.set_page_config(
    page_title="CreditGuard AI | Dashboard Overview",
    page_icon="🛡️",
    layout="wide",
)

PURPLE = "#a855f7"
PINK = "#ec4899"
FRAUD = "#fb7185"
NORMAL = "#34d399"


def apply_theme(theme: str) -> None:
    dark = theme == "Dark Mode"
    colors = {
        "bg": "#11162a" if dark else "#f7f5fc",
        "surface": "#1c2440" if dark else "#ffffff",
        "surface_alt": "#222c4a" if dark else "#fffaff",
        "text": "#f4f1ff" if dark else "#25233a",
        "muted": "#a9b1ca" if dark else "#77748d",
        "border": "#303b5d" if dark else "#ebe5f3",
        "shadow": "rgba(3, 7, 18, .22)" if dark else "rgba(92, 57, 130, .08)",
        "grid": "rgba(208, 198, 226, .18)" if dark else "rgba(112, 93, 145, .13)",
    }
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {colors["bg"]}; color: {colors["text"]}; }}
        [data-testid="stSidebar"] {{
            background: {colors["surface"]};
            border-right: 1px solid {colors["border"]};
        }}
        .block-container {{ max-width: 1600px; padding-top: 1.7rem; padding-bottom: 2.5rem; }}
        .cg-eyebrow {{ color: {PURPLE}; font-size: .72rem; font-weight: 800; letter-spacing: .17em; }}
        .cg-subtitle {{ color: {colors["muted"]}; margin-top: -.55rem; }}
        .cg-hero {{
            display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
            padding: 1.35rem 1.6rem; margin: .5rem 0 1.25rem;
            border: 1px solid {colors["border"]}; border-radius: 20px;
            background: linear-gradient(112deg, {colors["surface"]} 0%, {colors["surface_alt"]} 72%, {"#36224e" if dark else "#fbe9f7"} 100%);
            box-shadow: 0 12px 30px {colors["shadow"]};
        }}
        .cg-hero-kicker {{ color: {PINK}; font-size: .72rem; font-weight: 800; letter-spacing: .12em; }}
        .cg-hero-title {{ color: {colors["text"]}; font-size: 1.45rem; font-weight: 750; margin: .35rem 0 .25rem; }}
        .cg-hero-copy {{ color: {colors["muted"]}; font-size: .9rem; }}
        .cg-hero-chip {{
            flex: 0 0 auto; border: 1px solid {colors["border"]}; border-radius: 999px;
            padding: .55rem .85rem; color: {PINK}; background: {"rgba(236,72,153,.10)" if dark else "rgba(236,72,153,.07)"};
            font-size: .7rem; letter-spacing: .08em; font-weight: 800;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-radius: 18px;
            border-color: {colors["border"]};
            box-shadow: 0 8px 22px {colors["shadow"]};
        }}
        div[data-testid="stPlotlyChart"] {{
            border: 1px solid {colors["border"]}; border-radius: 16px;
            padding: 8px; background: {colors["surface"]};
            box-shadow: 0 8px 22px {colors["shadow"]};
        }}
        .cg-rail-label {{ color: {colors["muted"]}; font-size: .78rem; }}
        .cg-health-row {{ padding: .35rem 0; border-bottom: 1px solid {colors["border"]}; }}
        .cg-health-row:last-child {{ border-bottom: 0; }}
.cg-health-item {{ display:flex; align-items:center; justify-content:space-between; gap:.75rem; padding:.7rem 0; border-bottom:1px solid {colors["border"]}; }}
.cg-health-item:last-child {{ border-bottom:0; }}
.cg-health-value {{ color:{colors["text"]}; font-size:1.05rem; font-weight:750; }}
.cg-status-pill {{ border-radius:999px; padding:.2rem .55rem; font-size:.72rem; font-weight:700; white-space:nowrap; }}
.cg-status-ok {{ color:#34d399; background:rgba(52,211,153,.13); }}
.cg-status-warn {{ color:#fbbf24; background:rgba(251,191,36,.13); }}
.cg-status-detected {{ color:#60a5fa; background:rgba(96,165,250,.13); }}
        @media (max-width: 900px) {{
            .cg-hero {{ align-items: flex-start; flex-direction: column; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def money(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.2f}"


with st.sidebar:
    st.markdown("# 🛡️ CreditGuard AI")
    st.caption("Fraud Detection Platform")
    st.markdown("---")
    uploaded = st.file_uploader("Upload transaction dataset", type=["csv"])
    st.caption("Choose the persistent dataset in Settings & Dataset, or upload a CSV for this session.")

apply_theme("Dark Mode")

settings = load_dataset_settings()
df: pd.DataFrame | None = None
load_error: str | None = None
source_label: str | None = None
if uploaded is not None:
    try:
        df = read_uploaded_csv(uploaded.getvalue())
        source_label = f"Uploaded for this session: {uploaded.name}"
    except Exception as exc:
        load_error = f"Could not read this CSV: {exc}"
else:
    active_dataset_path = resolve_dataset_path(settings)
    if active_dataset_path is not None:
        try:
            df = load_dataset(active_dataset_path)
            source_label = f"Auto-loaded from {active_dataset_path.relative_to(active_dataset_path.parents[1]).as_posix()}"
        except Exception as exc:
            load_error = f"Could not read {active_dataset_path.name}: {exc}"

if df is not None and df.empty:
    load_error = "The selected CSV has no rows."
    df = None

data = df.copy() if df is not None else pd.DataFrame()
mapping = resolve_column_mapping([str(column) for column in data.columns], settings)
label_col = mapping["target"]
amount_col = mapping["amount"]
time_col = mapping["time"]
id_col = mapping["transaction_id"]
positive_class = settings.get("positive_class")
if label_col and not data.empty:
    available_targets = target_values(data[label_col])
    if positive_class not in available_targets:
        positive_class = infer_positive_class(available_targets)
labels = map_target_labels(data[label_col], positive_class) if label_col and not data.empty else pd.Series(dtype="object")
total = len(data)
fraud_count = int((labels == "Fraud").sum()) if total else 0
normal_count = int((labels == "Normal").sum()) if total else 0
rate = (fraud_count / total * 100) if total else None

st.markdown('<div class="cg-eyebrow">OVERVIEW / COMMAND CENTER</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <section class="cg-hero">
      <div>
        <div class="cg-hero-kicker">CREDITGUARD AI · FRAUD ANALYTICS</div>
        <div class="cg-hero-title">Dashboard Overview</div>
        <div class="cg-hero-copy">A real-time overview of transaction activity and fraud detection performance.</div>
      </div>
      <div class="cg-hero-chip">{"DATASET CONNECTED" if df is not None else "NO DATASET CONNECTED"}</div>
    </section>
    """,
    unsafe_allow_html=True,
)
if load_error:
    st.error(load_error, title="Dataset could not be loaded")

metrics = [
    ("Total Transactions", f"{total:,}" if total else "—", "Rows in uploaded or local dataset"),
    ("Fraud Transactions", f"{fraud_count:,}" if total and label_col else "—", "Actual fraud labels"),
    ("Normal Transactions", f"{normal_count:,}" if total and label_col else "—", "Actual normal labels"),
    ("Fraud Rate", f"{rate:.3f}%" if rate is not None and label_col else "—", "Fraud labels ÷ total rows"),
]
metric_cols = st.columns(4, gap="small")
for col, (label, value, note) in zip(metric_cols, metrics):
    col.metric(label, value, help=note, border=True)

if df is None:
    st.info("Upload a CSV in the sidebar or add data/transactions.csv / data/creditcard.csv to populate the dashboard.")
elif label_col is None:
    st.warning("No recognized target label column was found. Class counts and label-based charts are unavailable.")

dist_col, health_col = st.columns([1.15, 1], gap="medium")
with dist_col:
    with st.container(border=True):
        st.subheader("Transaction Distribution")
        if total and label_col:
            distribution = labels.value_counts().rename_axis("Actual Label").reset_index(name="Transactions")
            fig = px.pie(
                distribution,
                names="Actual Label",
                values="Transactions",
                hole=0.68,
                color="Actual Label",
                color_discrete_map={"Normal": NORMAL, "Fraud": FRAUD},
            )
            fig.update_traces(textposition="inside", textinfo="percent", hole=0.68)
            fig.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", y=-0.06),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig,
                width="stretch",
                alt="Donut chart of transaction counts by actual dataset label",
            )
        else:
            st.caption("Class distribution appears after a labeled dataset is available.")

with health_col:
    with st.container(border=True):
        st.subheader("Dataset Health")
        if total:
            missing = int(data.isna().sum().sum())
            duplicates = int(data.duplicated().sum())
            checks = [
                ("Missing Values", missing, "Clean" if missing == 0 else "Needs review", missing == 0, "ok" if missing == 0 else "warn"),
                ("Duplicate Rows", duplicates, "Clean" if duplicates == 0 else "Needs review", duplicates == 0, "ok" if duplicates == 0 else "warn"),
                ("Feature Columns", len(data.columns), "Detected", len(data.columns) > 0, "detected"),
            ]
            for name, value, status, healthy, state_style in checks:
                st.markdown(
                    f'<div class="cg-health-item"><div><strong>{name}</strong><br>'
                    f'<span class="cg-status-pill cg-status-{state_style}">{status}</span></div>'
                    f'<span class="cg-health-value">{value:,}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Health checks will run after a dataset is available.")

amount_col_view, summary_col = st.columns([1.55, 1], gap="medium")
with amount_col_view:
    with st.container(border=True):
        st.subheader("Transaction Amount Overview")
        if total and amount_col:
            amounts = pd.to_numeric(data[amount_col], errors="coerce")
            chart_df = pd.DataFrame({"Amount": amounts, "Actual Label": labels}).dropna(subset=["Amount"])
            upper_limit = chart_df["Amount"].quantile(0.99)
            outside_count = int((chart_df["Amount"] > upper_limit).sum())
            visible_df = chart_df[chart_df["Amount"] <= upper_limit]
            fig = px.histogram(
                visible_df,
                x="Amount",
                color="Actual Label",
                barmode="overlay",
                opacity=0.78,
                color_discrete_map={"Normal": NORMAL, "Fraud": FRAUD},
                nbins=40,
            )
            fig.update_layout(
                height=340,
                margin=dict(t=18, b=40, l=55, r=24),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Actual Label",
                xaxis_title="Transaction Amount",
                yaxis_title="Transactions",
                xaxis=dict(gridcolor="rgba(130,130,150,.18)", automargin=True),
                yaxis=dict(gridcolor="rgba(130,130,150,.18)", automargin=True),
            )
            st.plotly_chart(
                fig,
                width="stretch",
                alt="Histogram of transaction amount by actual dataset label, through the 99th percentile",
            )
            if outside_count:
                st.caption(
                    f"For readability, the histogram shows amounts through {upper_limit:,.2f}; "
                    f"{outside_count:,} rows above that range are omitted from this chart only."
                )
        else:
            st.caption("A recognized Amount column is required for this chart.")

with summary_col:
    with st.container(border=True):
        st.subheader("Class Summary")
        if total and label_col:
            counts = labels.value_counts()
            summary = pd.DataFrame(
                {
                    "Actual Label": counts.index,
                    "Transactions": counts.values,
                    "Share (%)": [f"{value:.3f}%" for value in counts.values / total * 100],
                }
            )
            st.table(summary)
        else:
            st.caption("Actual class counts appear after a labeled dataset is available.")

with st.container(border=True):
    st.subheader("Recent Transactions")
    st.caption("Rows and labels are taken from the uploaded dataset. Model predictions are not shown.")
    if total:
        recent = data.tail(10).copy().iloc[::-1]
        display = pd.DataFrame(index=recent.index)
        display["Transaction ID"] = recent[id_col].values if id_col else [str(i) for i in recent.index]
        display["Time"] = recent[time_col].values if time_col else "—"
        display["Amount"] = recent[amount_col].map(money).values if amount_col else "—"
        display["Actual Label"] = labels.loc[recent.index].values if label_col else "Not available"
        st.dataframe(
            display.reset_index(drop=True),
            hide_index=True,
            width="stretch",
            alt="Latest rows from the uploaded dataset with actual labels",
        )
    else:
        st.caption("Recent transactions appear here after a CSV is uploaded.")





