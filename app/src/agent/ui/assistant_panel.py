"""CreditGuard AI Assistant panel for Streamlit pages."""

from __future__ import annotations

import json
import streamlit as st


def _inject_styles(page_key: str) -> None:
    panel_key = f"cg_assistant_panel_{page_key}"
    tab_key = f"cg_assistant_tab_{page_key}"

    st.markdown(
        f"""
        <style>
        /* The assistant remains a real Streamlit column, so the page reflows. */
        .st-key-{panel_key} {{
            position: sticky;
            top: .5rem;
            height: calc(100vh - 1rem);
            max-height: calc(100vh - 1rem);
            overflow-y: auto;
            box-sizing: border-box;
            padding: 1rem .8rem 1rem 1rem;
            margin-right: -1rem;
            width: calc(100% + 1rem);
            border: 1px solid rgba(148,163,184,.22);
            border-right: 0;
            border-radius: 16px 0 0 16px;
            background: #171b2b;
            box-shadow: -8px 0 24px rgba(0,0,0,.14);
        }}
        .st-key-{panel_key} [data-testid="stVerticalBlock"] {{
            gap: .7rem;
        }}
        .st-key-{panel_key} [data-testid="stChatMessage"] {{
            border-radius: 13px;
            padding: .65rem .75rem;
            background: rgba(255,255,255,.045);
        }}
        .st-key-{panel_key} [data-testid="stChatMessage"] p {{
            line-height: 1.55;
        }}
        .st-key-{panel_key} [data-testid="stTextInput"] input {{
            border-radius: 10px;
        }}
        .st-key-{panel_key} button {{
            white-space: nowrap;
        }}
        .cg-assistant-subtitle {{
            color: #a7afc4;
            font-size: .8rem;
            margin-top: -.55rem;
        }}
        .cg-assistant-welcome {{
            color: #e7e9f2;
            font-size: .9rem;
            line-height: 1.6;
        }}
        .st-key-{tab_key} {{
            position: fixed;
            z-index: 99999;
            right: 0;
            top: 42%;
            width: 40px;
        }}
        .st-key-{tab_key} button {{
            min-height: 145px;
            width: 40px;
            padding: .45rem .25rem;
            border: 1px solid rgba(148,163,184,.35);
            border-right: 0;
            border-radius: 11px 0 0 11px;
            background: #252a40;
            color: #f4f1ff;
            font-size: .66rem;
            font-weight: 750;
            letter-spacing: .1em;
            writing-mode: vertical-rl;
            transform: rotate(180deg);
        }}
        @media (max-width: 900px) {{
            .st-key-{panel_key} {{
                position: static;
                height: auto;
                max-height: none;
                width: 100%;
                margin-right: 0;
                border-right: 1px solid rgba(148,163,184,.22);
                border-radius: 14px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def assistant_layout(page_key: str):
    """Create a responsive two-column layout when the assistant is open."""
    state_key = f"cg_assistant_open_{page_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    _inject_styles(page_key)

    if st.session_state[state_key]:
        main_col, assistant_col = st.columns([0.70, 0.30], gap="small")
        with assistant_col:
            panel = st.container(key=f"cg_assistant_panel_{page_key}")
        return main_col, panel

    with st.container(key=f"cg_assistant_tab_{page_key}"):
        if st.button(
            "CREDITGUARD AI",
            key=f"cg_open_assistant_{page_key}",
            help="Buka CreditGuard AI Assistant",
        ):
            st.session_state[state_key] = True
            st.rerun()

    return st.container(), None


def _get_nested(data: dict, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _welcome_message(page_key: str, context: dict) -> str:
    """Create the familiar contextual greeting without inventing unavailable values."""
    page_label = {
        "monitoring": "Transaction Monitoring",
        "investigation": "Fraud Investigation",
    }.get(page_key, page_key.replace("_", " ").title())

    # Support both a direct agent result and a context wrapper.
    result = context
    for wrapper in ("agent_result", "system_result", "analysis_result", "result"):
        wrapped = context.get(wrapper) if isinstance(context, dict) else None
        if isinstance(wrapped, dict) and any(
            key in wrapped for key in ("ml_prediction", "risk_assessment", "decision")
        ):
            result = wrapped
            break

    transaction_name = (
        context.get("transaction_name")
        or context.get("transaction_id")
        or context.get("transaction_reference")
        or "transaksi yang dipilih"
    )
    ml = result.get("ml_prediction", {}) if isinstance(result, dict) else {}
    risk = result.get("risk_assessment", {}) if isinstance(result, dict) else {}
    decision = result.get("decision", {}) if isinstance(result, dict) else {}

    lines = [
        f"Halo! Saya **CreditGuard AI Assistant**.",
        f"Berdasarkan data dari halaman **{page_label}** untuk **{transaction_name}**, berikut ringkasan informasi sistem yang tersedia:",
    ]

    def add_line(label, value):
        if value is not None and value != "":
            lines.append(f"- **{label}:** {value}")

    if isinstance(ml, dict):
        probability = ml.get("fraud_probability")
        if probability is not None:
            try:
                add_line("Probabilitas Fraud", f"{float(probability) * 100:.4f}%")
            except (TypeError, ValueError):
                add_line("Probabilitas Fraud", probability)
        add_line("Status Prediksi", ml.get("classification"))

    if isinstance(risk, dict):
        add_line("Tingkat Risiko (Risk Level)", risk.get("risk_level"))

    if isinstance(decision, dict):
        add_line(
            "Rekomendasi Tindakan",
            decision.get("recommended_action") or decision.get("action"),
        )

    if len(lines) == 2:
        lines.append(
            "Silakan jalankan analisis atau tanyakan hal yang ingin dipahami. "
            "Saya akan menjelaskan berdasarkan konteks yang tersedia."
        )
    else:
        lines.append(
            "\nSilakan tanyakan jika ingin memahami hasil prediksi, tingkat risiko, "
            "atau rekomendasi sistem lebih lanjut."
        )

    return "\n\n".join(lines)


def _build_prompt(page_key: str, context: dict, history: list[dict], question: str) -> str:
    safe_context = json.dumps(context, ensure_ascii=False, default=str)
    conversation = "\n".join(
        f'{item["role"].upper()}: {item["content"]}' for item in history[-8:]
    )
    return f"""
Kamu adalah CreditGuard AI Assistant, asisten penjelasan untuk aplikasi deteksi fraud kartu kredit.
Halaman aktif: {page_key}.
Konteks halaman dan hasil sistem (sumber fakta):
{safe_context}

Riwayat percakapan:
{conversation or "(belum ada)"}

Pertanyaan pengguna:
{question}

Aturan:
- Jawab dalam Bahasa Indonesia dengan jelas dan kontekstual.
- Gunakan fakta dari konteks yang tersedia; jangan mengarang informasi transaksi.
- Jangan menghitung ulang atau mengubah probabilitas fraud, threshold, tingkat risiko, atau rekomendasi sistem.
- Jangan mengklaim melakukan pemblokiran atau tindakan nyata pada sistem perbankan.
- Jika informasi tidak tersedia, sampaikan dengan jujur.
- Prediksi berasal dari model ML dan keputusan berasal dari policy deterministik; Gemini memberi penjelasan pendukung.
""".strip()


def render_assistant_panel(page_key: str, context: dict | None = None) -> None:
    """Render contextual chat. Session history remains after closing the panel."""
    context = context or {}
    history_key = f"cg_assistant_history_{page_key}"
    history = st.session_state.setdefault(history_key, [])
    welcome_key = f"cg_assistant_welcome_added_{page_key}"

    header_col, close_col = st.columns([0.86, 0.14], gap="small")
    with header_col:
        st.markdown("### CreditGuard AI Assistant")
        st.markdown(
            '<div class="cg-assistant-subtitle">Asisten penjelasan kontekstual · Gemini</div>',
            unsafe_allow_html=True,
        )
    with close_col:
        if st.button("×", key=f"cg_close_assistant_{page_key}", help="Tutup"):
            st.session_state[f"cg_assistant_open_{page_key}"] = False
            st.rerun()

    # Add the original-style greeting once per page session, not on every rerun.
    if not st.session_state.get(welcome_key):
        history.append(
            {"role": "assistant", "content": _welcome_message(page_key, context)}
        )
        st.session_state[welcome_key] = True

    st.divider()

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form(key=f"cg_assistant_form_{page_key}", clear_on_submit=True):
        question = st.text_input(
            "Pesan",
            placeholder="Tanyakan tentang halaman atau hasil analisis...",
            label_visibility="collapsed",
            key=f"cg_assistant_question_{page_key}",
        )
        submitted = st.form_submit_button("Kirim  ↑", use_container_width=True)

    if submitted and question.strip():
        clean_question = question.strip()
        st.session_state[history_key].append({"role": "user", "content": clean_question})
        try:
            from src.agent.llm.gemini_client import generate_response

            prompt = _build_prompt(
                page_key,
                context,
                st.session_state[history_key][:-1],
                clean_question,
            )
            with st.spinner("CreditGuard AI sedang menyiapkan jawaban..."):
                answer = generate_response(prompt)
            st.session_state[history_key].append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.session_state[history_key].append(
                {
                    "role": "assistant",
                    "content": (
                        "Maaf, layanan Gemini sedang mengalami kendala sehingga "
                        "belum dapat menjawab. Hasil prediksi dan keputusan sistem "
                        "tetap tersedia. Silakan coba kembali beberapa saat lagi."
                    ),
                }
            )
        st.rerun()
