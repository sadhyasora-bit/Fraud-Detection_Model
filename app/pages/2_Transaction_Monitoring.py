"""User-facing Transaction Monitoring page for CreditGuard AI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import load_dataset_settings, load_dataset, resolve_dataset_path
from src.model_inference import load_fraud_model
from src.agent.orchestrator import run_agent_with_assessment
from src.agent.ui.assistant_panel import assistant_layout, render_assistant_panel

st.set_page_config(page_title="CreditGuard AI | Transaction Monitoring", page_icon="🛡️", layout="wide")

# Policy settings remain internal; users do not need to configure model thresholds.
DEFAULT_THRESHOLD = 0.50
DEFAULT_ANALYSIS_TRIGGER = 0.10
DEFAULT_REVIEW_THRESHOLD = 0.20
DEFAULT_ESCALATE_THRESHOLD = 0.70


PURPLE = "#a855f7"
PINK = "#ec4899"


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
    }

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {colors["bg"]};
            color: {colors["text"]};
        }}

        [data-testid="stSidebar"] {{
            background: {colors["surface"]};
            border-right: 1px solid {colors["border"]};
        }}

        .block-container {{
            max-width: 1600px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }}

        .cg-eyebrow {{
            color: {PURPLE};
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .17em;
        }}

        .cg-hero {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.35rem 1.6rem;
            margin: .5rem 0 1.25rem;
            border: 1px solid {colors["border"]};
            border-radius: 20px;
            background: linear-gradient(
                112deg,
                {colors["surface"]} 0%,
                {colors["surface_alt"]} 72%,
                {"#36224e" if dark else "#fbe9f7"} 100%
            );
            box-shadow: 0 12px 30px {colors["shadow"]};
        }}

        .cg-hero-kicker {{
            color: {PINK};
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .12em;
        }}

        .cg-hero-title {{
            color: {colors["text"]};
            font-size: 1.45rem;
            font-weight: 750;
            margin: .35rem 0 .25rem;
        }}

        .cg-hero-copy {{
            color: {colors["muted"]};
            font-size: .9rem;
        }}

        .cg-hero-chip {{
            flex: 0 0 auto;
            border: 1px solid {colors["border"]};
            border-radius: 999px;
            padding: .55rem .85rem;
            color: {PINK};
            background: {"rgba(236,72,153,.10)" if dark else "rgba(236,72,153,.07)"};
            font-size: .7rem;
            letter-spacing: .08em;
            font-weight: 800;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-radius: 18px;
            border-color: {colors["border"]};
            box-shadow: 0 8px 22px {colors["shadow"]};
        }}

        @media (max-width: 900px) {{
            .cg-hero {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme("Dark Mode")



@st.cache_data(show_spinner=False)
def get_model_features() -> list[str]:
    model = load_fraud_model()
    return list(model.feature_names_in_)


def prepare_transaction(dataframe: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame) or len(dataframe) != 1:
        raise ValueError("Exactly one transaction must be selected.")
    missing = [feature for feature in expected_features if feature not in dataframe.columns]
    if missing:
        raise ValueError("The selected transaction is missing required information.")
    transaction = dataframe.loc[:, expected_features].copy()
    for column in expected_features:
        transaction[column] = pd.to_numeric(transaction[column], errors="coerce")
    if transaction.isna().any().any() or not np.isfinite(transaction.to_numpy(dtype=float)).all():
        raise ValueError("The selected transaction contains incomplete or invalid values.")
    return transaction.reset_index(drop=True)


def find_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}
    return next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)


def humanize(value) -> str:
    return str(value).replace("_", " ").replace("-", " ").strip().title()


def render_transaction_details(dataframe: pd.DataFrame, row_index: int) -> None:
    id_col = find_column(dataframe, ["Transaction ID", "Transaction_ID", "transactionid", "ID", "id"])
    amount_col = find_column(dataframe, ["Amount", "Transaction Amount", "amount"])
    time_col = find_column(dataframe, ["Transaction Time", "Timestamp", "Datetime", "Date", "Time"])
    type_col = find_column(dataframe, ["Transaction Type", "Type", "Transaction_Type"])
    row = dataframe.iloc[row_index]

    fields = [
        ("Transaction ID", str(row[id_col])) if id_col is not None else
        ("Transaction Reference", f"TRX-{row_index + 1:06d}")
    ]
    if amount_col is not None:
        try:
            amount = float(row[amount_col])
            formatted_amount = f"Rp {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (TypeError, ValueError):
            formatted_amount = str(row[amount_col])

        fields.append(("Amount", formatted_amount))
        
    if time_col is not None:
        try:
            time_seconds = float(row[time_col])

            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = int(time_seconds % 60)

            formatted_time = (
                f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            )

        except (TypeError, ValueError):
            formatted_time = str(row[time_col])

        fields.append(
            ("Elapsed Time", formatted_time)
        )
    if type_col is not None:
        fields.append(("Transaction Type", str(row[type_col])))

    with st.container(border=True):
        st.markdown("**Selected Transaction**")
        cols = st.columns(min(4, len(fields)))
        for index, (label, value) in enumerate(fields):
            cols[index].metric(label, value)


def generate_explanation(result: dict) -> dict:
    ml = result["ml_prediction"]
    risk = result["risk_assessment"]
    analysis = result["transaction_analysis"]
    action = str(result["decision"]["recommended_action"]).upper()
    policy = result["policy_configuration"]
    probability = float(ml["fraud_probability"])
    risk_level = str(risk["risk_level"]).upper()
    review = float(policy["review_threshold"])
    escalate = float(policy["escalate_threshold"])

    if risk_level == "LOW":
        interpretation = "Transaksi ini memiliki indikasi risiko fraud yang rendah berdasarkan hasil pemeriksaan sistem."
    elif risk_level == "MEDIUM":
        interpretation = "Transaksi ini menunjukkan indikasi risiko yang perlu diperhatikan dan disarankan untuk ditinjau lebih lanjut."
    else:
        interpretation = "Transaksi ini menunjukkan indikasi risiko fraud yang tinggi dan disarankan untuk segera ditindaklanjuti."

    evidence = [f"Estimasi kemungkinan fraud sebesar {probability:.2%}."]
    if probability < review:
        evidence.append("Hasil estimasi masih berada di bawah batas pemeriksaan.")
    elif probability < escalate:
        evidence.append("Hasil estimasi telah mencapai tingkat yang memerlukan pemeriksaan.")
    else:
        evidence.append("Hasil estimasi telah mencapai tingkat eskalasi.")

    if analysis.get("analysis_skipped"):
        evidence.append("Pemeriksaan indikator tambahan tidak diperlukan berdasarkan hasil penyaringan awal.")
    else:
        indicators = analysis.get("transaction_indicators") or {}
        active = [humanize(name) for name, value in indicators.items() if bool(value)]
        if active:
            evidence.append("Indikator tambahan yang terdeteksi: " + ", ".join(active) + ".")
        elif indicators:
            evidence.append("Tidak ada indikator tambahan yang terdeteksi.")
        if analysis.get("amount_category"):
            evidence.append("Kategori nominal transaksi: " + humanize(analysis["amount_category"]) + ".")
        if analysis.get("time_category"):
            evidence.append("Kategori waktu transaksi: " + humanize(analysis["time_category"]) + ".")

    action_text = {
        "ALLOW": "Sistem merekomendasikan transaksi dapat dilanjutkan berdasarkan tingkat risiko yang terukur.",
        "REVIEW": "Sistem merekomendasikan pemeriksaan manual sebelum keputusan operasional dibuat.",
        "ESCALATE": "Sistem merekomendasikan transaksi diteruskan untuk investigasi lebih lanjut.",
    }.get(action, "Sistem belum dapat memberikan rekomendasi tindakan.")

    return {
        "probability": probability, "risk_level": risk_level, "action": action,
        "interpretation": interpretation, "evidence": evidence, "action_text": action_text,
        "analysis": analysis,
    }


def render_agent_assessment(assessment_result: dict | None) -> None:
    """Show Gemini's separate assessment without changing system decisions."""
    st.markdown("### AI Agent Assessment")

    if not isinstance(assessment_result, dict):
        st.info("Penilaian AI Agent belum tersedia. Hasil sistem tetap menjadi acuan.")
        return

    if assessment_result.get("status") == "success":
        with st.container(border=True):
            st.markdown(assessment_result.get("assessment", "Penilaian tidak berisi teks."))
            st.caption("Penilaian pendukung dari Gemini; tidak mengubah hasil ML, tingkat risiko, atau rekomendasi sistem.")
    else:
        fallback = assessment_result.get(
            "fallback_message",
            "Penilaian AI Agent sedang tidak tersedia. Hasil sistem tetap dapat digunakan.",
        )
        st.warning(fallback)


def render_result(result: dict) -> None:
    if result.get("status") != "success":
        st.error("Analisis belum berhasil diselesaikan. Silakan coba kembali atau hubungi administrator sistem.")
        return

    info = generate_explanation(result)
    probability, risk_level, action = info["probability"], info["risk_level"], info["action"]

    st.markdown("### Analysis Result")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Fraud Probability", f"{probability:.2%}")
        c2.metric("Risk Level", risk_level.title())
        st.progress(min(max(probability, 0.0), 1.0), text="Estimated fraud probability")
        st.markdown("**Assessment**")
        st.write(info["interpretation"])

    st.markdown("### Why Was This Transaction Flagged?")
    with st.container(border=True):
        for item in info["evidence"]:
            st.markdown(f"- {item}")

    st.markdown("### Recommended Action")
    with st.container(border=True):
        if action == "ALLOW":
            st.success("**ALLOW · Low estimated risk**")
        elif action == "REVIEW":
            st.warning("**REVIEW · Manual review recommended**")
        else:
            st.error("**ESCALATE · Further investigation recommended**")
        st.write(info["action_text"])

    if not info["analysis"].get("analysis_skipped"):
        indicators = info["analysis"].get("transaction_indicators") or {}
        active = [humanize(name) for name, value in indicators.items() if bool(value)]
        if active:
            st.markdown("### Additional Findings")
            with st.container(border=True):
                for item in active:
                    st.markdown(f"- {item}")

    st.caption(
        "CreditGuard AI memberikan rekomendasi berdasarkan hasil analisis model. "
        "Sistem tidak secara otomatis menyetujui, menolak, atau memblokir transaksi."
    )


# Right-side AI Assistant layout
main_col, assistant_col = assistant_layout("monitoring")

with main_col:
    # Page header — mengikuti hero header Dashboard Overview.
    st.markdown(
        '<div class="cg-eyebrow">TRANSACTION MONITORING / FRAUD DETECTION</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="cg-hero">
          <div>
            <div class="cg-hero-kicker">CREDITGUARD AI · FRAUD ANALYTICS</div>
            <div class="cg-hero-title">Transaction Monitoring</div>
            <div class="cg-hero-copy">
              Monitor transaksi, tinjau estimasi risiko fraud, dan pahami rekomendasi tindak lanjut.
            </div>
          </div>
          <div class="cg-hero-chip">TRANSACTION ANALYSIS</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Pilih transaksi untuk diperiksa dan dapatkan penilaian risiko beserta rekomendasi tindak lanjut."
    )

    # Load model and configured dataset
    try:
        expected_features = get_model_features()
    except Exception:
        st.error("Layanan analisis belum dapat dimuat. Silakan hubungi administrator sistem.")
        st.stop()

    settings = load_dataset_settings()
    dataset_path = resolve_dataset_path(settings)
    if dataset_path is None:
        st.warning("Belum ada sumber transaksi yang aktif. Silakan minta administrator mengatur dataset melalui Settings & Dataset.")
        st.stop()

    try:
        dataset = load_dataset(dataset_path)
    except Exception:
        st.error("Data transaksi tidak dapat dibuka. Silakan periksa konfigurasi dataset atau hubungi administrator.")
        st.stop()

    if dataset.empty:
        st.warning("Sumber data transaksi belum memiliki data.")
        st.stop()

    missing = [feature for feature in expected_features if feature not in dataset.columns]
    if missing:
        st.error("Struktur data belum sesuai dengan kebutuhan analisis. Silakan hubungi administrator sistem.")
        st.stop()

    # Transaction selection
    st.markdown("### Select Transaction")
    id_col = find_column(dataset, ["Transaction ID", "Transaction_ID", "transactionid", "ID", "id"])
    options = list(range(len(dataset)))

    if id_col is not None:
        ids = dataset[id_col].astype(str)
        selected_row = st.selectbox(
            "Find a transaction", options,
            format_func=lambda index: ids.iloc[index],
            key="monitoring_selected_row",
            help="Pilih transaksi berdasarkan ID yang tersedia pada dataset.",
        )
    else:
        selected_row = st.selectbox(
            "Choose a transaction", options,
            format_func=lambda index: f"Transaction {index + 1:06d}",
            key="monitoring_selected_row",
            help="ID transaksi tidak tersedia. Referensi ditampilkan berdasarkan urutan data.",
        )

    render_transaction_details(dataset, int(selected_row))
    transaction_data = dataset.iloc[[int(selected_row)]].copy()

    # Run analysis
    st.markdown("### Fraud Detection")
    st.markdown("Jalankan pemeriksaan untuk mengetahui estimasi risiko dan rekomendasi tindakan terhadap transaksi yang dipilih.")

    if st.button("Analyze Transaction", type="primary", width="stretch"):
        try:
            prepared = prepare_transaction(transaction_data, expected_features)
            with st.spinner("CreditGuard AI sedang memeriksa transaksi..."):
                agent_response = run_agent_with_assessment(
                    transaction=prepared,
                    threshold=DEFAULT_THRESHOLD,
                    analysis_trigger=DEFAULT_ANALYSIS_TRIGGER,
                    review_threshold=DEFAULT_REVIEW_THRESHOLD,
                    escalate_threshold=DEFAULT_ESCALATE_THRESHOLD,
                    assessment_timeout=40,
                )

            # Keep the deterministic result separate from Gemini's assessment.
            system_result = agent_response.get("system_result")
            if not isinstance(system_result, dict):
                system_result = agent_response

            st.session_state["last_agent_result"] = system_result
            st.session_state["last_agent_assessment"] = agent_response.get("agent_assessment")
            st.session_state["last_agent_decision_modified"] = agent_response.get("decision_modified", False)
            st.session_state["last_transaction_row"] = int(selected_row)
            st.session_state["last_transaction_label"] = (
                str(dataset.iloc[int(selected_row)][id_col]) if id_col is not None
                else f"Transaction {int(selected_row) + 1:06d}"
            )
        except Exception:
            st.error("Transaksi belum dapat dianalisis. Pastikan data valid dan coba kembali.")

    # Show only the result for the currently selected transaction
    last_result = st.session_state.get("last_agent_result")
    last_row = st.session_state.get("last_transaction_row")
    if last_result is not None and last_row == int(selected_row):
        st.divider()
        st.markdown("### Transaction Assessment")
        st.caption("Transaction: " + str(st.session_state.get("last_transaction_label", "Selected transaction")))
        render_result(last_result)
        render_agent_assessment(st.session_state.get("last_agent_assessment"))
    elif last_result is not None:
        st.info("Pilih Analyze Transaction untuk memeriksa transaksi yang saat ini dipilih.")


if assistant_col is not None:
    with assistant_col:
        _result = st.session_state.get("last_agent_result")
        _assistant_context = {
            "page": "Transaction Monitoring",
            "selected_transaction": st.session_state.get("last_transaction_label", "Belum ada transaksi dianalisis"),
            "analysis_status": _result.get("status") if isinstance(_result, dict) else "not_run",
            "fraud_probability": (_result.get("ml_prediction", {}).get("fraud_probability") if isinstance(_result, dict) else None),
            "risk_level": (_result.get("risk_assessment", {}).get("risk_level") if isinstance(_result, dict) else None),
            "recommended_action": (_result.get("decision", {}).get("recommended_action") if isinstance(_result, dict) else None),
            "transaction_analysis": (_result.get("transaction_analysis", {}) if isinstance(_result, dict) else None),
            "agent_assessment": st.session_state.get("last_agent_assessment"),
            "decision_modified": st.session_state.get("last_agent_decision_modified", False),
        }
        render_assistant_panel("monitoring", _assistant_context)
