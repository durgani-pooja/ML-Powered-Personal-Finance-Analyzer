# app.py — main Streamlit app
# This is the brain that connects all the core modules
# and displays everything in a clean dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.categorizer import load_classifier, categorize_transactions
from core.anomaly import detect_anomalies
from core.forecaster import forecast_spending
from core.advisor import get_saving_advice

# ── Page Setup ────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Analyzer",
    page_icon="💰",
    layout="wide"
)

# Load custom CSS
import os
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("💰 ML-Powered Personal Finance Analyzer")
st.caption("Upload your bank statement → get instant insights, anomaly alerts, and AI saving tips.")

# ── Cache heavy model so it only loads once ───────────────
@st.cache_resource
def get_classifier():
    with st.spinner("Loading NLP model (first time only, ~30 seconds)..."):
        return load_classifier()

# ── Sidebar — File Upload ─────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Statement")
    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Needs columns: Date, Description, Amount"
    )
    st.markdown("---")
    st.markdown("**Sample CSV format:**")
    st.code("Date,Description,Amount\n2024-01-05,Swiggy,450\n2024-01-15,Salary,-65000")
    st.caption("Tip: Negative amounts = income (salary credit)")

# ── Main App ──────────────────────────────────────────────
if uploaded is not None:

    # Load and validate CSV
    try:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df = df.dropna(subset=["Amount"])
    except Exception as e:
        st.error(f"Couldn't read file: {e}")
        st.stop()

    # ── Step 1: Categorize ────────────────────────────────
    with st.spinner("🔍 Categorizing transactions with AI..."):
        classifier = get_classifier()
        df = categorize_transactions(df, classifier)

    # ── Step 2: Detect Anomalies ──────────────────────────
    df = detect_anomalies(df)

    # ── Key Metrics Row ───────────────────────────────────
    expenses = df[df["Amount"] > 0]
    income = df[df["Amount"] < 0]["Amount"].abs().sum()
    total_spent = expenses["Amount"].sum()
    savings = income - total_spent
    anomaly_count = df["Is_Anomaly"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💸 Total Spent", f"₹{total_spent:,.0f}")
    col2.metric("💵 Total Income", f"₹{income:,.0f}")
    col3.metric("🏦 Savings", f"₹{savings:,.0f}",
                delta=f"{(savings/income*100):.1f}% saved" if income > 0 else None)
    col4.metric("⚠️ Anomalies", int(anomaly_count),
                delta="unusual spends" if anomaly_count > 0 else "all normal",
                delta_color="inverse")

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🍩 Spending by Category")
        cat_summary = expenses.groupby("Category")["Amount"].sum().reset_index()
        fig_pie = px.pie(
            cat_summary,
            values="Amount",
            names="Category",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("📅 Daily Spending Over Time")
        daily = expenses.groupby("Date")["Amount"].sum().reset_index()
        fig_line = px.line(
            daily, x="Date", y="Amount",
            labels={"Amount": "Spent (₹)"},
            color_discrete_sequence=["#FF6600"]
        )
        fig_line.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ── Anomaly Table ─────────────────────────────────────
    st.subheader("⚠️ Unusual Spending Detected")
    anomalies = df[df["Is_Anomaly"] == True][["Date", "Description", "Amount", "Category"]]
    if len(anomalies) > 0:
        st.dataframe(
            anomalies.style.format({"Amount": "₹{:,.0f}"}),
            use_container_width=True
        )
    else:
        st.success("No unusual transactions found — your spending looks consistent!")

    st.markdown("---")

    # ── Forecast ──────────────────────────────────────────
    st.subheader("🔮 30-Day Spending Forecast")
    with st.spinner("Forecasting next month..."):
        forecast = forecast_spending(df)

    if not forecast.empty:
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            name="Predicted Spend",
            line=dict(color="#FF6600", width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,102,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Range"
        ))
        fig_forecast.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily Spend (₹)",
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        predicted_monthly = forecast["yhat"].sum()
        st.info(f"📊 Estimated spend next 30 days: **₹{predicted_monthly:,.0f}**")
    else:
        st.warning("Need more transaction history for forecasting (at least 10 days).")

    st.markdown("---")

    # ── AI Saving Tips ────────────────────────────────────
    st.subheader("🤖 Personalised Saving Tips (Powered by LLaMA-3)")
    if st.button("✨ Generate My Saving Tips", type="primary"):
        with st.spinner("Asking LLaMA-3 for advice..."):
            tips = get_saving_advice(df)
        st.markdown(f"""
        <div class="tip-box">
        {tips.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Full Transaction Table ────────────────────────────
    with st.expander("📋 View All Transactions"):
        st.dataframe(
            df[["Date", "Description", "Amount", "Category", "Is_Anomaly"]]
            .style.format({"Amount": "₹{:,.0f}"}),
            use_container_width=True
        )

else:
    # Landing page when no file uploaded
    st.info("👈 Upload your bank statement CSV from the sidebar to get started.")

    st.markdown("### What this app does:")
    col1, col2, col3, col4 = st.columns(4)
    col1.success("🏷️ **Auto-categorizes** every transaction using NLP")
    col2.warning("⚠️ **Detects anomalies** — unusual spending spikes")
    col3.info("🔮 **Forecasts** your next 30 days of expenses")
    col4.success("🤖 **AI tips** — personalised saving advice via LLaMA-3")
