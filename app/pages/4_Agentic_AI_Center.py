"""CreditGuard AI — Agentic AI Center (audit-log observability)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CreditGuard AI | Agentic AI Center", page_icon="🤖", layout="wide")

# Visual system aligned with Dashboard Overview
PURPLE, PINK = "#a855f7", "#ec4899"
GREEN, RED, AMBER = "#34d399", "#fb7185", "#fbbf24"
BG, SURFACE, ALT = "#11162a", "#1c2440", "#222c4a"
TEXT, MUTED, BORDER = "#f4f1ff", "#a9b1ca", "#303b5d"

st.markdown(f"""
<style>
.stApp{{background:{BG};color:{TEXT}}}
[data-testid="stSidebar"]{{background:{SURFACE};border-right:1px solid {BORDER}}}
.block-container{{max-width:1600px;padding-top:1.55rem;padding-bottom:2.5rem}}
.cg-eyebrow{{color:{PURPLE};font-size:.72rem;font-weight:850;letter-spacing:.17em}}
.cg-hero{{display:flex;justify-content:space-between;align-items:center;gap:1.5rem;
padding:1.45rem 1.65rem;margin:.45rem 0 1.2rem;border:1px solid {BORDER};border-radius:20px;
background:linear-gradient(112deg,{SURFACE} 0%,{ALT} 72%,#36224e 100%);
box-shadow:0 12px 30px rgba(3,7,18,.22)}}
.cg-kicker{{color:{PINK};font-size:.7rem;font-weight:850;letter-spacing:.13em}}
.cg-title{{color:{TEXT};font-size:1.55rem;font-weight:800;margin:.35rem 0}}
.cg-copy{{color:{MUTED};font-size:.9rem}}
.cg-chip{{border:1px solid {BORDER};border-radius:999px;padding:.58rem .9rem;color:{GREEN};
background:rgba(52,211,153,.10);font-size:.7rem;font-weight:850;letter-spacing:.08em;white-space:nowrap}}
[data-testid="stVerticalBlockBorderWrapper"]>div{{border-radius:17px;border-color:{BORDER};box-shadow:0 8px 22px rgba(3,7,18,.2)}}
[data-testid="stMetric"]{{background:{SURFACE};border:1px solid {BORDER};border-radius:15px;padding:13px 15px}}
[data-testid="stMetricLabel"] p{{color:{MUTED}!important}}
div[data-testid="stPlotlyChart"]{{border:1px solid {BORDER};border-radius:15px;padding:7px;background:{SURFACE}}}
.cg-step{{border:1px solid {BORDER};border-radius:15px;padding:15px 9px;min-height:108px;
background:linear-gradient(145deg,{SURFACE},{ALT});text-align:center}}
.cg-no{{color:{PINK};font-size:.68rem;font-weight:850;letter-spacing:.12em}}
.cg-step-title{{color:{TEXT};font-weight:750;margin-top:9px;font-size:.88rem}}
.cg-note{{color:{MUTED};font-size:.72rem;margin-top:5px}}
.cg-muted{{color:{MUTED};font-size:.82rem}}
@media(max-width:900px){{.cg-hero{{align-items:flex-start;flex-direction:column}}}}
</style>
""", unsafe_allow_html=True)

APP_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = APP_ROOT / "data" / "logs" / "agent_audit.jsonl"
WORKFLOW = [
    ("ml_prediction", "ML Prediction", "Fraud probability"),
    ("threshold_analysis", "Threshold Analysis", "Classification threshold"),
    ("transaction_analysis", "Transaction Analysis", "Conditional analysis"),
    ("risk_assessment", "Risk Assessment", "Risk level and rationale"),
    ("decision", "Decision", "Recommended action"),
]
LABELS = {x[0]: x[1] for x in WORKFLOW}
ACTION_COLORS = {"ALLOW": GREEN, "REVIEW": AMBER, "ESCALATE": RED, "NOT GENERATED": MUTED}

@st.cache_data(show_spinner=False)
def load_records(path: str, modified: float) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except (json.JSONDecodeError, TypeError):
                pass
    return rows

def timestamp(value):
    if not value:
        return None
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else result

def display_time(value):
    value = timestamp(value)
    return value.tz_convert("Asia/Jakarta").strftime("%d %b %Y · %H:%M:%S WIB") if value is not None else "N/A"

def build_executions(records):
    grouped = defaultdict(list)
    for row in records:
        if row.get("run_id"):
            grouped[row["run_id"]].append(row)
    result = []
    for run_id, events in grouped.items():
        events.sort(key=lambda x: str(x.get("timestamp") or ""))
        outputs = {}
        for e in events:
            if e.get("status") != "failed" and isinstance(e.get("output"), dict):
                outputs[e.get("tool_name")] = e["output"]
        decision = outputs.get("decision", {})
        failed = any(e.get("status") == "failed" for e in events)
        state = "FAILED" if failed else ("COMPLETED" if decision else "INCOMPLETE")
        prediction, risk = outputs.get("ml_prediction", {}), outputs.get("risk_assessment", {})
        probability = prediction.get("fraud_probability")
        try:
            probability = float(probability) if probability is not None else None
        except (TypeError, ValueError):
            probability = None
        parsed = [timestamp(e.get("timestamp")) for e in events]
        parsed = [x for x in parsed if x is not None]
        duration = (max(parsed) - min(parsed)).total_seconds() if len(parsed) > 1 else 0
        result.append({
            "run_id": run_id, "timestamp": events[-1].get("timestamp"),
            "parsed": timestamp(events[-1].get("timestamp")), "status": state,
            "risk": str(risk.get("risk_level", "UNKNOWN")).upper(),
            "action": str(decision.get("recommended_action", "NOT GENERATED")).upper(),
            "probability": probability, "events": events, "outputs": outputs,
            "event_count": len(events),
            "success": sum(e.get("status") == "success" for e in events),
            "failed": sum(e.get("status") == "failed" for e in events),
            "skipped": sum(e.get("event_type") == "tool_skipped" for e in events),
            "duration": max(0, duration),
        })
    return sorted(result, key=lambda x: x["parsed"] or pd.Timestamp.min.tz_localize("UTC"), reverse=True)

def chart_style(fig, height=300):
    fig.update_layout(height=height, margin=dict(t=15,b=25,l=15,r=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(208,198,226,.15)")
    fig.update_yaxes(gridcolor="rgba(208,198,226,.15)")
    return fig

# Header
st.markdown('<div class="cg-eyebrow">AUTONOMOUS SYSTEM / OBSERVABILITY</div>', unsafe_allow_html=True)
st.markdown("""
<section class="cg-hero">
 <div><div class="cg-kicker">CREDITGUARD AI · AGENT OPERATIONS</div>
 <div class="cg-title">Agentic AI Center</div>
 <div class="cg-copy">Observe Agent executions, inspect tool-level activity, and understand how fraud-risk recommendations are produced.</div></div>
 <div class="cg-chip">AUDIT MONITORING</div>
</section>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 Agent Center")
    st.caption("Execution observability")
    st.markdown("---")
    st.caption("Audit source")
    st.code("data/logs/agent_audit.jsonl", language="text")
    st.caption("This page monitors existing runs. It does not start a new Agent execution.")

if not LOG_PATH.exists():
    st.info("Audit log belum tersedia. Jalankan analisis dari Transaction Monitoring terlebih dahulu.")
    st.stop()

records = load_records(str(LOG_PATH), LOG_PATH.stat().st_mtime)
executions = build_executions(records)
a, b = st.columns([1, 4])
with a:
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with b:
    st.markdown(f'<div class="cg-muted" style="padding:.55rem 0">Source: <b>{LOG_PATH.name}</b> · {len(records):,} audit records · {len(executions):,} executions with run_id</div>', unsafe_allow_html=True)

if not records:
    st.info("Belum ada audit event. Jalankan analisis melalui Transaction Monitoring.")
    st.stop()
if not executions:
    st.warning("Audit records tersedia, tetapi belum ada run_id. Legacy records tidak dapat dikelompokkan menjadi eksekusi.")
    st.stop()

# Filters
st.markdown("#### Explore executions")
f1, f2, f3, f4 = st.columns([1, 1, 1.15, 1.7])
with f1: sf = st.selectbox("Status", ["All", "COMPLETED", "FAILED", "INCOMPLETE"])
with f2: rf = st.selectbox("Risk", ["All", "LOW", "MEDIUM", "HIGH", "UNKNOWN"])
with f3: af = st.selectbox("Recommendation", ["All", "ALLOW", "REVIEW", "ESCALATE", "NOT GENERATED"])
with f4: query = st.text_input("Find execution by Run ID", placeholder="Paste a Run ID")
view = executions
if sf != "All": view = [x for x in view if x["status"] == sf]
if rf != "All": view = [x for x in view if x["risk"] == rf]
if af != "All": view = [x for x in view if x["action"] == af]
if query.strip(): view = [x for x in view if query.lower() in x["run_id"].lower()]
if not view:
    st.warning("Tidak ada eksekusi yang sesuai filter.")
    st.stop()

# KPIs
st.markdown("#### Execution health")
total = len(view)
completed = sum(x["status"] == "COMPLETED" for x in view)
failed = sum(x["status"] == "FAILED" for x in view)
incomplete = sum(x["status"] == "INCOMPLETE" for x in view)
events_total = sum(x["event_count"] for x in view)
success_total = sum(x["success"] for x in view)
k = st.columns(5, gap="small")
for col, label, value, help_text in [
    (k[0], "Total executions", total, "Filtered Agent runs"),
    (k[1], "Completed", completed, f"{completed/total*100:.1f}% of filtered runs"),
    (k[2], "Failed", failed, "Runs with a failed tool event"),
    (k[3], "Incomplete", incomplete, "No final decision event"),
    (k[4], "Tool events", events_total, f"{success_total} successful events"),
]:
    col.metric(label, f"{value:,}", help=help_text, border=True)

# Pipeline
with st.container(border=True):
    st.subheader("Agent execution pipeline")
    st.caption("The actual sequence implemented by src/agent/orchestrator.py.")
    cols = st.columns(5, gap="small")
    for i, (_, name, note) in enumerate(WORKFLOW):
        with cols[i]:
            st.markdown(f'<div class="cg-step"><div class="cg-no">STAGE {i+1:02d}</div><div class="cg-step-title">{name}</div><div class="cg-note">{note}</div></div>', unsafe_allow_html=True)
    st.caption("Transaction Analysis is conditional. It may be recorded as skipped when fraud probability is below analysis_trigger; skipped is not a tool failure.")

# Charts
left, right = st.columns([1.2, 1], gap="medium")
with left:
    with st.container(border=True):
        st.subheader("Execution activity")
        st.caption("Daily count based on each run's last recorded event.")
        activity = pd.DataFrame([{"Date": x["parsed"], "Status": x["status"], "Runs": 1} for x in view if x["parsed"] is not None])
        if not activity.empty:
            activity["Date"] = activity["Date"].dt.tz_convert("Asia/Jakarta").dt.date
            activity = activity.groupby(["Date", "Status"], as_index=False)["Runs"].sum()
            fig = px.bar(activity, x="Date", y="Runs", color="Status", barmode="stack",
                color_discrete_map={"COMPLETED": GREEN, "FAILED": RED, "INCOMPLETE": AMBER})
            fig.update_layout(xaxis_title=None, yaxis_title="Executions", legend_title=None)
            st.plotly_chart(chart_style(fig), use_container_width=True)
        else: st.caption("Timestamp valid tidak tersedia.")
with right:
    with st.container(border=True):
        st.subheader("Recommendation mix")
        st.caption("Distribution of final decision outputs.")
        counts = Counter(x["action"] for x in view)
        d = pd.DataFrame([{"Recommendation": key, "Executions": value} for key, value in counts.items()])
        if not d.empty:
            fig = px.pie(d, names="Recommendation", values="Executions", hole=.66,
                color="Recommendation", color_discrete_map=ACTION_COLORS)
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(legend=dict(orientation="h", y=-.08))
            st.plotly_chart(chart_style(fig), use_container_width=True)
        else: st.caption("Decision belum tercatat.")

# Tool performance table
with st.container(border=True):
    st.subheader("Tool activity & reliability")
    st.caption("Counts are calculated from recorded tool events. Skipped events are separated from failures.")
    rows = []
    for key, label, _ in WORKFLOW:
        ev = [e for x in view for e in x["events"] if e.get("tool_name") == key]
        ok = sum(e.get("status") == "success" and e.get("event_type") != "tool_skipped" for e in ev)
        bad = sum(e.get("status") == "failed" for e in ev)
        skip = sum(e.get("event_type") == "tool_skipped" for e in ev)
        rows.append({"Agent tool": label, "Events": len(ev), "Successful": ok,
                     "Failed": bad, "Skipped": skip,
                     "Success rate*": f"{ok/(ok+bad)*100:.1f}%" if ok+bad else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("*Success rate excludes skipped events and is not the same as ML model accuracy.")

# History
with st.container(border=True):
    st.subheader("Recent Agent executions")
    st.caption("Each row represents one Agent run_id, not a transaction ID.")
    history = []
    for x in view:
        history.append({
            "Last activity (WIB)": (timestamp(x["timestamp"]).tz_convert("Asia/Jakarta").strftime("%d %b %Y %H:%M:%S") if timestamp(x["timestamp"]) is not None else "N/A"),
            "Run ID": x["run_id"], "Status": x["status"], "Risk": x["risk"],
            "Fraud probability": f"{x['probability']:.2%}" if x["probability"] is not None else "N/A",
            "Recommendation": x["action"], "Tool events": x["event_count"],
            "Duration (s)": f"{x['duration']:.3f}",
        })
    st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)

# Drill-down
with st.container(border=True):
    st.subheader("Execution inspector")
    st.caption("Select a run to inspect its outputs and chronological tool audit trail.")
    run_ids = [x["run_id"] for x in view]
    selected_id = st.selectbox("Select Agent execution", run_ids,
        format_func=lambda rid: f"{rid[:8]}… · {next(x['status'] for x in view if x['run_id']==rid)} · {next(x['action'] for x in view if x['run_id']==rid)}")
    selected = next(x for x in view if x["run_id"] == selected_id)
    p = selected["probability"]
    q = st.columns(4)
    q[0].metric("Execution status", selected["status"])
    q[1].metric("Risk level", selected["risk"])
    q[2].metric("Recommendation", selected["action"])
    q[3].metric("Fraud probability", f"{p:.2%}" if p is not None else "N/A")
    st.caption(f"Run ID: `{selected_id}` · Last activity: {display_time(selected['timestamp'])} · Duration: {selected['duration']:.3f}s")
    out1, out2 = st.columns(2)
    with out1:
        st.markdown("**Risk assessment**")
        st.json(selected["outputs"].get("risk_assessment", {"message": "No risk assessment output recorded."}))
    with out2:
        st.markdown("**Final decision**")
        st.json(selected["outputs"].get("decision", {"message": "No final decision recorded."}))
    st.markdown("#### Tool execution audit trail")
    for i, event in enumerate(selected["events"], start=1):
        tool = event.get("tool_name", "unknown")
        event_type, status = event.get("event_type", "unknown"), event.get("status", "unknown")
        icon = "🔴" if status == "failed" else ("⏭️" if event_type == "tool_skipped" else "🟢")
        with st.expander(f"{icon} {i:02d} · {LABELS.get(tool, tool)} · {event_type} · {status}"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.caption(f"Time: {display_time(event.get('timestamp'))}")
                st.caption(f"Status: {status}")
            with c2:
                st.markdown("**Input**"); st.json(event.get("input", {}))
                st.markdown("**Output**"); st.json(event.get("output", {}))

with st.expander("Data interpretation & limitations"):
    st.markdown("""
    - **Execution** adalah satu pemanggilan `run_agent()` yang dikelompokkan dengan `run_id`.
    - **Tool event** adalah satu catatan aktivitas tool di dalam eksekusi.
    - **Completed** berarti event `decision` tercatat; **Failed** berarti terdapat event berstatus gagal; selain itu **Incomplete**.
    - **Transaction Analysis** dapat dilewati berdasarkan `analysis_trigger`; skipped bukan kegagalan.
    - Agent menghasilkan rekomendasi. Implementasi saat ini tidak otomatis memblokir transaksi atau menjalankan tindakan eksternal.
    - `run_id` bukan `Transaction ID`; audit log saat ini belum menjamin identitas transaksi sumber tersimpan.
    - Success rate tool bukan model accuracy. Evaluasi akurasi model berada di Model Performance.
    - Semua metrik dan grafik hanya merangkum audit records yang tersedia setelah filter diterapkan.
    """)
