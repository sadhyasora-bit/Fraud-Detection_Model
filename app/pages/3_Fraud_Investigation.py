
"""Fraud Investigation page for CreditGuard AI."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.ui.assistant_panel import assistant_layout, render_assistant_panel


st.set_page_config(
    page_title="CreditGuard AI | Fraud Investigation",
    page_icon="🔎",
    layout="wide",
)


# ==================================================
# THEME
# ==================================================

PURPLE = "#a855f7"
PINK = "#ec4899"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: #11162a;
            color: #f4f1ff;
        }}

        [data-testid="stSidebar"] {{
            background: #1c2440;
            border-right: 1px solid #303b5d;
        }}

        .block-container {{
            max-width: 1600px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}

        [data-testid="stVerticalBlock"] {{
            gap: .75rem;
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
            padding: 1.2rem 1.5rem;
            margin: .4rem 0 1rem;
            border: 1px solid #303b5d;
            border-radius: 18px;
            background: linear-gradient(
                112deg, #1c2440 0%, #222c4a 72%, #36224e 100%
            );
            box-shadow: 0 10px 24px rgba(3, 7, 18, .20);
        }}

        .cg-hero-kicker {{
            color: {PINK};
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .12em;
        }}

        .cg-hero-title {{
            color: #f4f1ff;
            font-size: 1.4rem;
            font-weight: 750;
            margin: .3rem 0 .2rem;
        }}

        .cg-hero-copy {{
            color: #a9b1ca;
            font-size: .88rem;
        }}

        .cg-hero-chip {{
            flex: 0 0 auto;
            border: 1px solid #303b5d;
            border-radius: 999px;
            padding: .5rem .8rem;
            color: {PINK};
            background: rgba(236, 72, 153, .10);
            font-size: .68rem;
            letter-spacing: .08em;
            font-weight: 800;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-radius: 15px;
            border-color: #303b5d;
            box-shadow: 0 6px 18px rgba(3, 7, 18, .16);
        }}

        [data-testid="stMetric"] {{
            padding: .2rem;
        }}

        [data-testid="stMetricLabel"] {{
            font-size: .82rem;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.65rem;
        }}

        h2 {{
            margin-top: .35rem !important;
            margin-bottom: .35rem !important;
        }}

        h3, h4 {{
            margin-top: .2rem !important;
            margin-bottom: .4rem !important;
        }}

        hr {{
            margin: .55rem 0 !important;
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


apply_theme()


# ==================================================
# PATHS
# ==================================================

APP_ROOT = Path(__file__).resolve().parents[1]

AUDIT_LOG_PATH = (
    APP_ROOT / "data" / "logs" / "agent_audit.jsonl"
)


# ==================================================
# LOAD AUDIT RECORDS
# ==================================================

@st.cache_data(show_spinner=False)
def load_audit_records() -> list[dict]:

    if not AUDIT_LOG_PATH.exists():
        return []

    records = []

    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if isinstance(record, dict):
                    records.append(record)

            except json.JSONDecodeError:
                continue

    return records


# ==================================================
# BUILD INVESTIGATION CASES
# ==================================================

def build_investigation_cases(records: list[dict]) -> list[dict]:

    grouped = defaultdict(list)

    for record in records:

        run_id = record.get("run_id")

        if run_id:
            grouped[run_id].append(record)

    cases = []

    for run_id, events in grouped.items():

        events.sort(
            key=lambda item: item.get("timestamp") or ""
        )

        case = {
            "run_id": run_id,
            "timestamp": events[-1].get("timestamp"),
            "ml_prediction": {},
            "risk_assessment": {},
            "decision": {},
            "transaction_analysis": {},
            "events": events,
        }

        for event in events:

            if event.get("status") == "failed":
                continue

            tool_name = event.get("tool_name")
            output = event.get("output", {})

            if not isinstance(output, dict):
                continue

            if tool_name == "ml_prediction":
                case["ml_prediction"] = output

            elif tool_name == "risk_assessment":
                case["risk_assessment"] = output

            elif tool_name == "decision":
                case["decision"] = output

            elif tool_name == "transaction_analysis":
                case["transaction_analysis"] = output

        decision = case["decision"]

        if not decision:
            continue

        case["recommended_action"] = decision.get(
            "recommended_action", "UNKNOWN"
        )

        case["risk_level"] = case["risk_assessment"].get(
            "risk_level", "UNKNOWN"
        )

        case["fraud_probability"] = case["ml_prediction"].get(
            "fraud_probability"
        )

        case["transaction_amount"] = case[
            "risk_assessment"
        ].get("transaction_amount")

        cases.append(case)

    cases.sort(
        key=lambda item: item.get("timestamp") or "",
        reverse=True,
    )

    return cases


# ==================================================
# FORMATTING HELPERS
# ==================================================

def format_probability(value) -> str:

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2%}"

    except (TypeError, ValueError):
        return "N/A"


def format_amount(value) -> str:

    if value is None:
        return "N/A"

    try:
        amount = float(value)

        return (
            f"Rp {amount:,.2f}"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

    except (TypeError, ValueError):
        return "N/A"


def format_timestamp(value) -> str:

    if not value:
        return "N/A"

    try:
        timestamp = pd.to_datetime(value)

        if pd.isna(timestamp):
            return str(value)

        return timestamp.strftime("%d %b %Y, %H:%M:%S")

    except (TypeError, ValueError):
        return str(value)


def display_value(value) -> str:

    if value is None:
        return "Belum tersedia"

    if isinstance(value, bool):
        return "Ya" if value else "Tidak"

    if isinstance(value, list):
        return ", ".join(map(str, value))

    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {item}"
            for key, item in value.items()
        )

    return str(value)


def get_action_label(action: str) -> str:

    labels = {
        "ALLOW": "Tidak Memerlukan Tindakan",
        "REVIEW": "Perlu Ditinjau",
        "ESCALATE": "Prioritas Investigasi",
    }

    return labels.get(action, "Belum Ditentukan")


def get_risk_label(level: str) -> str:

    labels = {
        "LOW": "Rendah",
        "MEDIUM": "Sedang",
        "HIGH": "Tinggi",
    }

    return labels.get(level, "Belum Diketahui")


def get_classification_label(value) -> str:

    labels = {
        "FRAUD": "Terindikasi Fraud",
        "NORMAL": "Normal",
        "LEGITIMATE": "Normal",
    }

    if value is None:
        return "Belum tersedia"

    return labels.get(str(value).upper(), str(value))


def get_category_label(value) -> str:

    labels = {
        "LOW": "Rendah",
        "MEDIUM": "Sedang",
        "HIGH": "Tinggi",
        "EARLY": "Awal periode",
        "NORMAL": "Normal",
        "LATE": "Akhir periode",
    }

    if value is None:
        return "Belum tersedia"

    return labels.get(str(value).upper(), str(value))


def get_reason_label(value: str) -> str:

    if not value:
        return "Belum tersedia."

    reasons = {
        "Fraud probability reached the review threshold.":
            "Probabilitas fraud telah mencapai batas yang memerlukan peninjauan.",

        "Transaction requires further investigation.":
            "Transaksi memerlukan pemeriksaan lebih lanjut.",
    }

    return reasons.get(value, value)


def get_event_label(tool_name: str) -> str:

    labels = {
        "ml_prediction": "Pemeriksaan pola transaksi",
        "threshold_analysis": "Evaluasi tingkat kecurigaan",
        "transaction_analysis": "Analisis karakteristik transaksi",
        "risk_assessment": "Penilaian tingkat risiko",
        "decision": "Penyusunan rekomendasi",
    }

    return labels.get(tool_name, "Tahapan pemeriksaan")


def render_indicators(value) -> None:

    if not value:
        st.caption("Tidak ada indikator tambahan yang tercatat.")
        return

    if isinstance(value, dict):

        for key, item in value.items():

            label = str(key).replace("_", " ").title()

            st.markdown(
                f"**{label}**: {display_value(item)}"
            )

    elif isinstance(value, list):

        for item in value:
            st.markdown(f"- {display_value(item)}")

    else:
        st.write(display_value(value))


# Right-side AI Assistant layout
main_col, assistant_col = assistant_layout("investigation")

with main_col:
    # ==================================================
    # PAGE HEADER
    # ==================================================

    st.markdown(
        '<div class="cg-eyebrow">MONITORING / INVESTIGATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="cg-hero">
            <div>
                <div class="cg-hero-kicker">
                    CREDITGUARD AI · FRAUD ANALYTICS
                </div>
                <div class="cg-hero-title">Fraud Investigation</div>
                <div class="cg-hero-copy">
                    Tinjau transaksi mencurigakan, pahami tingkat risiko,
                    dan periksa rekomendasi tindak lanjut.
                </div>
            </div>
            <div class="cg-hero-chip">INVESTIGATION CENTER</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


    # ==================================================
    # LOAD DATA
    # ==================================================

    try:
        audit_records = load_audit_records()

    except Exception:
        st.error(
            "Data pemeriksaan belum dapat dimuat. "
            "Silakan coba kembali."
        )
        st.stop()


    if not audit_records:

        st.info(
            "Belum ada riwayat pemeriksaan. "
            "Lakukan pemeriksaan transaksi melalui halaman "
            "Transaction Monitoring terlebih dahulu."
        )

        st.stop()


    cases = build_investigation_cases(audit_records)

    if not cases:

        st.info(
            "Belum ada hasil pemeriksaan yang dapat ditampilkan."
        )

        st.stop()


    # ==================================================
    # INVESTIGATION OVERVIEW
    # ==================================================

    st.subheader("Investigation Overview")

    total_analyses = len(cases)

    allow_analyses = sum(
        case["recommended_action"] == "ALLOW"
        for case in cases
    )

    review_analyses = sum(
        case["recommended_action"] == "REVIEW"
        for case in cases
    )

    escalated_analyses = sum(
        case["recommended_action"] == "ESCALATE"
        for case in cases
    )

    overview_metrics = [
        ("Total Pemeriksaan", total_analyses),
        ("Tidak Memerlukan Tindakan", allow_analyses),
        ("Perlu Ditinjau", review_analyses),
        ("Prioritas Investigasi", escalated_analyses),
    ]

    metric_cols = st.columns(4, gap="small")

    for col, (label, value) in zip(metric_cols, overview_metrics):

        with col:
            with st.container(border=True):
                st.metric(label, f"{value:,}")


    # ==================================================
    # INVESTIGATION QUEUE
    # ==================================================

    st.divider()
    st.subheader("Investigation Queue")

    investigation_cases = [
        case
        for case in cases
        if case["recommended_action"] in ["REVIEW", "ESCALATE"]
    ]

    st.caption(
        "Daftar transaksi yang membutuhkan peninjauan "
        "atau pemeriksaan lebih lanjut."
    )


    # Assign stable display references before applying filters.
    case_references = {
        case["run_id"]: f"INV-{index:04d}"
        for index, case in enumerate(investigation_cases, start=1)
    }


    # ==================================================
    # FILTERS
    # ==================================================

    filter_col1, filter_col2 = st.columns(2, gap="small")

    with filter_col1:

        action_filter = st.selectbox(
            "Rekomendasi Tindak Lanjut",
            ["Semua", "Perlu Ditinjau", "Prioritas Investigasi"],
        )

    with filter_col2:

        risk_filter = st.selectbox(
            "Tingkat Risiko",
            ["Semua", "Rendah", "Sedang", "Tinggi"],
        )


    action_mapping = {
        "Perlu Ditinjau": "REVIEW",
        "Prioritas Investigasi": "ESCALATE",
    }

    risk_mapping = {
        "Rendah": "LOW",
        "Sedang": "MEDIUM",
        "Tinggi": "HIGH",
    }

    filtered_cases = investigation_cases.copy()

    if action_filter != "Semua":

        filtered_cases = [
            case for case in filtered_cases
            if case["recommended_action"] == action_mapping[action_filter]
        ]

    if risk_filter != "Semua":

        filtered_cases = [
            case for case in filtered_cases
            if case["risk_level"] == risk_mapping[risk_filter]
        ]


    # ==================================================
    # QUEUE TABLE
    # ==================================================

    if not filtered_cases:

        st.info(
            "Tidak ada kasus yang sesuai dengan filter yang dipilih."
        )

    else:

        queue_data = []

        for case in filtered_cases:

            queue_data.append({
                "Referensi Kasus": case_references[case["run_id"]],
                "Waktu Pemeriksaan": format_timestamp(case["timestamp"]),
                "Probabilitas Fraud": format_probability(
                    case["fraud_probability"]
                ),
                "Tingkat Risiko": get_risk_label(case["risk_level"]),
                "Rekomendasi": get_action_label(
                    case["recommended_action"]
                ),
                "Nominal Transaksi": format_amount(
                    case["transaction_amount"]
                ),
            })

        with st.container(border=True):

            st.dataframe(
                pd.DataFrame(queue_data),
                width="stretch",
                hide_index=True,
            )


    # ==================================================
    # INVESTIGATION DETAIL
    # ==================================================

    st.divider()
    st.subheader("Investigation Detail")

    if filtered_cases:

        case_options = {}

        for case in filtered_cases:

            reference = case_references[case["run_id"]]

            label = (
                f"{reference} · "
                f"{get_action_label(case['recommended_action'])} · "
                f"{format_timestamp(case['timestamp'])}"
            )

            case_options[label] = case

        selected_label = st.selectbox(
            "Pilih kasus untuk melihat detail",
            list(case_options.keys()),
        )

        selected_case = case_options[selected_label]

        ml = selected_case["ml_prediction"]
        risk = selected_case["risk_assessment"]
        decision = selected_case["decision"]
        analysis = selected_case["transaction_analysis"]

        # --------------------------------------------------
        # A. RINGKASAN PENILAIAN
        # --------------------------------------------------

        detail_col1, detail_col2, detail_col3 = st.columns(
            3,
            gap="small",
        )

        with detail_col1:

            with st.container(border=True):

                st.metric(
                    "Probabilitas Fraud",
                    format_probability(
                        selected_case["fraud_probability"]
                    ),
                )

        with detail_col2:

            with st.container(border=True):

                st.metric(
                    "Tingkat Risiko",
                    get_risk_label(selected_case["risk_level"]),
                )

        with detail_col3:

            with st.container(border=True):

                st.markdown(
                    "<div style='font-size:.82rem;color:#a9b1ca'>"
                    "Rekomendasi Tindak Lanjut</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{get_action_label(selected_case['recommended_action'])}**"
                )

        st.caption(
            f"Waktu pemeriksaan: "
            f"{format_timestamp(selected_case['timestamp'])}"
        )

        # --------------------------------------------------
        # B. INFORMASI TRANSAKSI
        # --------------------------------------------------

        detail_col1, detail_col2 = st.columns(
            2,
            gap="small",
        )

        with detail_col1:

            with st.container(border=True):

                st.markdown("#### Informasi Transaksi")

                st.caption("Nominal Transaksi")

                st.markdown(
                    f"### {format_amount(risk.get('transaction_amount'))}"
                )

                st.caption("Hasil Klasifikasi")

                st.markdown(
                    f"**{get_classification_label(ml.get('classification'))}**"
                )

        # --------------------------------------------------
        # C. HASIL PENILAIAN
        # --------------------------------------------------

        with detail_col2:

            with st.container(border=True):

                st.markdown("#### Hasil Penilaian")

                st.caption("Alasan Penilaian Risiko")

                st.write(
                    get_reason_label(risk.get("risk_reason"))
                )

                st.caption("Alasan Rekomendasi")

                st.write(
                    get_reason_label(decision.get("reason"))
                )

        # --------------------------------------------------
        # D. INDIKATOR TRANSAKSI
        # --------------------------------------------------

        if analysis and not analysis.get("analysis_skipped"):

            with st.container(border=True):

                st.markdown("#### Indikator Transaksi")

                indicator_col1, indicator_col2 = st.columns(2)

                with indicator_col1:

                    st.caption("Kategori Nominal")

                    st.markdown(
                        f"**{get_category_label(analysis.get('amount_category'))}**"
                    )

                with indicator_col2:

                    st.caption("Kategori Waktu")

                    st.markdown(
                        f"**{get_category_label(analysis.get('time_category'))}**"
                    )

                st.divider()

                st.markdown("**Indikator yang Ditemukan**")

                render_indicators(
                    analysis.get("transaction_indicators")
                )

        elif analysis and analysis.get("analysis_skipped"):

            st.info(
                "Analisis karakteristik transaksi tidak dilakukan "
                "pada pemeriksaan ini."
            )

        # --------------------------------------------------
        # E. REKOMENDASI TINDAK LANJUT
        # --------------------------------------------------

        action = selected_case["recommended_action"]

        with st.container(border=True):

            st.markdown("#### Rekomendasi Tindak Lanjut")

            if action == "ESCALATE":

                st.error(
                    "Kasus ini direkomendasikan untuk mendapatkan "
                    "prioritas pemeriksaan lebih lanjut."
                )

            elif action == "REVIEW":

                st.warning(
                    "Kasus ini memerlukan peninjauan lebih lanjut "
                    "sebelum keputusan berikutnya dibuat."
                )

            else:

                st.success(
                    "Transaksi tidak memerlukan tindakan lanjutan "
                    "berdasarkan hasil pemeriksaan sistem."
                )

            st.caption(
                "Hasil ini merupakan rekomendasi sistem berdasarkan "
                "analisis risiko, bukan tindakan otomatis untuk "
                "menyetujui, menolak, atau memblokir transaksi."
            )

    else:

        st.info(
            "Pilih filter yang sesuai untuk melihat detail kasus."
        )


    # ==================================================
    # INVESTIGATION HISTORY
    # ==================================================

    st.divider()
    st.subheader("Investigation History")

    st.caption(
        "Tahapan pemeriksaan yang dilakukan pada kasus terpilih."
    )

    if filtered_cases:

        selected_events = selected_case["events"]

        history_data = []

        for event in selected_events:

            tool_name = event.get("tool_name", "")

            history_data.append({
                "Tahapan Pemeriksaan": get_event_label(tool_name),
                "Waktu": format_timestamp(event.get("timestamp")),
                "Status": (
                    "Berhasil"
                    if event.get("status") != "failed"
                    else "Tidak Berhasil"
                ),
            })

        with st.container(border=True):

            if history_data:

                st.dataframe(
                    pd.DataFrame(history_data),
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.caption(
                    "Belum ada riwayat tahapan pemeriksaan."
                )

    else:

        st.info(
            "Pilih kasus untuk melihat riwayat pemeriksaannya."
        )


    # ==================================================
    # FOOTER
    # ==================================================

    st.divider()

    st.caption(
        "CreditGuard AI · Fraud Investigation"
    )


if assistant_col is not None:
    with assistant_col:
        _case = locals().get("selected_case")
        _assistant_context = {
            "page": "Fraud Investigation",
            "selected_case": ({
                "case_reference": case_references.get(_case.get("run_id"), "Selected case"),
                "fraud_probability": _case.get("fraud_probability"),
                "risk_level": _case.get("risk_level"),
                "recommended_action": _case.get("recommended_action"),
                "transaction_amount": _case.get("transaction_amount"),
                "risk_reason": _case.get("risk_assessment", {}).get("risk_reason"),
                "decision_reason": _case.get("decision", {}).get("reason"),
                "transaction_analysis": _case.get("transaction_analysis", {}),
            } if isinstance(_case, dict) else "Belum ada kasus yang dipilih"),
        }
        render_assistant_panel("investigation", _assistant_context)
