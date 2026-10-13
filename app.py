from __future__ import annotations

import requests
import streamlit as st

API = "http://localhost:8000"
st.set_page_config(page_title="UPI Shield", page_icon="🛡️", layout="wide")

st.title("🛡️ UPI Shield")
st.caption("Adaptive mule-account layering prevention prototype — simulated environment")

with st.sidebar:
    st.header("System Status")
    try:
        h = requests.get(f"{API}/health", timeout=3).json()
        if h.get("status") == "ok":
            st.success("API + model ready")
        else:
            st.warning(h.get("error", "Setup required"))
    except requests.RequestException:
        st.error("FastAPI is not running")

pre, post = st.columns(2)

with pre:
    st.subheader("1. Pre-Transaction Guard")
    receiver = st.text_input("Recipient UPI ID", "mule012@upi", key="receiver")
    amount = st.number_input("Amount (₹)", min_value=1.0, max_value=500000.0, value=15000.0, step=500.0)
    if st.button("Check Before PIN", type="primary", use_container_width=True):
        try:
            r = requests.post(f"{API}/api/v1/pre-check", json={"sender_upi": "victim@upi", "receiver_upi": receiver, "amount": amount}, timeout=10)
            r.raise_for_status()
            data = r.json()
            st.metric("Recipient Risk Index", f"{data['risk_score']:.1f}/100")
            st.metric("Crowd Reports", data["total_reports"])
            if data["warning_level"] == "RED_ALERT": st.error("🚨 HIGH RISK")
            elif data["warning_level"] == "AMBER": st.warning("⚠️ REVIEW")
            else: st.success("✅ LOW RISK")
            st.write(data["message"])
            st.json({k: data[k] for k in ["verified", "account_age_days", "recent_tx_count_5m", "report_categories"]})
        except Exception as exc:
            st.error(str(exc))

with post:
    st.subheader("2. Post-Transaction Layering Guard")
    outflow = st.number_input("Observed downstream outflow in 5 min (₹)", min_value=0.0, value=48500.0, step=1000.0)
    downstream = st.number_input("Distinct downstream accounts in 5 min", min_value=0, value=7, step=1)
    if st.button("Run AI + LangGraph Guardrail", use_container_width=True):
        try:
            r = requests.post(f"{API}/api/v1/post-check", json={"receiver_upi": receiver, "observed_outflow_5m": outflow, "downstream_accounts_5m": downstream}, timeout=20)
            r.raise_for_status()
            data = r.json()
            st.metric("Behavioral Risk", f"{data['risk_score']:.1f}/100")
            st.metric("Anomaly Probability", f"{data['anomaly_probability']:.2%}")
            if data["decision"] == "HOLD": st.error("⛔ LIEN / DOWNSTREAM HOLD")
            elif data["decision"] == "REVIEW": st.warning("🟠 MANUAL REVIEW / ENHANCED MONITORING")
            else: st.success("🟢 SAFE — NO LIEN")
            st.write(f"**Status:** {data['lien_status']}")
            st.write("**State transitions:**")
            for item in data["action_log"]:
                st.write(f"• {item}")
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.subheader("Demo Accounts")
st.write("Try `mule012@upi`, `mule025@upi`, or `user0001@upi` to demonstrate different risk profiles.")
