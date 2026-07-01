import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="FraudShield AI",
    page_icon="💳",
    layout="wide"
)


# ==============================
# CSS - Light Professional UI
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F6F8FC;
    color: #111827;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

[data-testid="stSidebar"] * {
    color: #111827 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #2563EB, #0F172A);
    padding: 34px;
    border-radius: 26px;
    color: white;
    box-shadow: 0px 15px 40px rgba(37, 99, 235, 0.25);
    margin-bottom: 28px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 17px;
    color: #DBEAFE;
}

.card {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 24px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 10px 30px rgba(15, 23, 42, 0.08);
    margin-bottom: 20px;
}

.kpi-card {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 24px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 10px 30px rgba(15, 23, 42, 0.08);
}

.kpi-label {
    color: #6B7280;
    font-size: 14px;
    font-weight: 600;
}

.kpi-value {
    color: #111827;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-foot {
    color: #6B7280;
    font-size: 13px;
    margin-top: 6px;
}

.blue-border { border-left: 7px solid #2563EB; }
.green-border { border-left: 7px solid #10B981; }
.red-border { border-left: 7px solid #EF4444; }
.orange-border { border-left: 7px solid #F97316; }

.alert-danger {
    background: #FEF2F2;
    border-left: 7px solid #EF4444;
    padding: 18px;
    border-radius: 18px;
    color: #991B1B;
    font-weight: 700;
    margin: 20px 0px;
}

.alert-success {
    background: #ECFDF5;
    border-left: 7px solid #10B981;
    padding: 18px;
    border-radius: 18px;
    color: #065F46;
    font-weight: 700;
    margin: 20px 0px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #111827;
    margin-top: 20px;
    margin-bottom: 12px;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #2563EB;
    border-radius: 18px;
    padding: 18px;
}

.stDownloadButton > button {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 1.2rem;
    font-weight: 800;
}

.stButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 800;
}

.sidebar-box {
    background: #F9FAFB;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join("models", "best_model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception:
    st.error("Model files not found. Make sure `best_model.pkl` and `scaler.pkl` are inside the `models` folder.")
    st.stop()


# ==============================
# Functions
# ==============================
def validate_input(data):
    required_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    return missing_columns, required_columns


def preprocess_data(data, required_columns):
    input_data = data[required_columns].copy()
    input_data[["Time", "Amount"]] = scaler.transform(input_data[["Time", "Amount"]])
    return input_data


def make_predictions(data):
    missing_columns, required_columns = validate_input(data)

    if missing_columns:
        return None, missing_columns

    processed_data = preprocess_data(data, required_columns)

    probabilities = model.predict_proba(processed_data)[:, 1]

    result_data = data.copy()
    result_data["Fraud_Probability"] = probabilities

    return result_data, None


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.markdown("## 💳 FraudShield AI")
    st.markdown("Enterprise-style fraud monitoring dashboard.")

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-box">
        <b>Required CSV columns</b><br><br>
        Time, V1–V28, Amount
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Detection Settings")

    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.caption("Lower threshold detects more fraud but may increase false alerts.")


# ==============================
# Header
# ==============================
st.markdown("""
<div class="hero">
    <div class="hero-title">FraudShield AI</div>
    <div class="hero-subtitle">
        Credit Card Fraud Detection Dashboard powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Upload transaction CSV file",
    type=["csv"]
)


if uploaded_file is None:
    st.markdown("""
    <div class="card">
        <h3>📂 Upload a transaction file to begin</h3>
        <p style="color:#6B7280;">
        This dashboard predicts whether credit card transactions are genuine or suspicious.
        Upload a CSV containing Time, V1–V28, and Amount columns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ==============================
# Load Uploaded File
# ==============================
try:
    df = pd.read_csv(uploaded_file)
except Exception:
    st.error("Unable to read the uploaded file. Please upload a valid CSV file.")
    st.stop()


results, missing_columns = make_predictions(df)

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


results["Prediction"] = (results["Fraud_Probability"] >= threshold).astype(int)
results["Prediction_Label"] = results["Prediction"].map({
    0: "Genuine",
    1: "Fraud"
})


# ==============================
# KPI Metrics
# ==============================
total_transactions = len(results)
fraud_count = int(results["Prediction"].sum())
genuine_count = total_transactions - fraud_count
fraud_percentage = (fraud_count / total_transactions) * 100
avg_fraud_probability = results["Fraud_Probability"].mean() * 100


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card blue-border">
        <div class="kpi-label">Total Transactions</div>
        <div class="kpi-value">{total_transactions:,}</div>
        <div class="kpi-foot">Uploaded records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card green-border">
        <div class="kpi-label">Genuine Transactions</div>
        <div class="kpi-value">{genuine_count:,}</div>
        <div class="kpi-foot">Predicted as safe</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card red-border">
        <div class="kpi-label">Suspicious Transactions</div>
        <div class="kpi-value">{fraud_count:,}</div>
        <div class="kpi-foot">Needs review</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card orange-border">
        <div class="kpi-label">Fraud Rate</div>
        <div class="kpi-value">{fraud_percentage:.2f}%</div>
        <div class="kpi-foot">Based on threshold</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# Alert
# ==============================
if fraud_count > 0:
    st.markdown(f"""
    <div class="alert-danger">
        🚨 Alert: {fraud_count} suspicious transaction(s) detected. Immediate analyst review recommended.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-success">
        ✅ No suspicious transactions detected at the selected threshold.
    </div>
    """, unsafe_allow_html=True)


# ==============================
# Charts
# ==============================
st.markdown('<div class="section-title">📊 Fraud Monitoring Overview</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    counts = results["Prediction_Label"].value_counts().reset_index()
    counts.columns = ["Prediction", "Count"]

    fig_pie = px.pie(
        counts,
        names="Prediction",
        values="Count",
        hole=0.55,
        color="Prediction",
        color_discrete_map={
            "Genuine": "#10B981",
            "Fraud": "#EF4444"
        }
    )

    fig_pie.update_layout(
        title="Transaction Classification",
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827")
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    fig_hist = px.histogram(
        results,
        x="Fraud_Probability",
        nbins=40,
        title="Fraud Probability Distribution",
        color_discrete_sequence=["#2563EB"]
    )

    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Threshold"
    )

    fig_hist.update_layout(
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827")
    )

    st.plotly_chart(fig_hist, use_container_width=True)


# ==============================
# Gauge + Amount Analysis
# ==============================
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_fraud_probability,
        title={"text": "Average Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [0, 30], "color": "#DCFCE7"},
                {"range": [30, 70], "color": "#FEF3C7"},
                {"range": [70, 100], "color": "#FEE2E2"}
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100
            }
        }
    ))

    fig_gauge.update_layout(
        height=420,
        paper_bgcolor="white",
        font=dict(color="#111827")
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

with chart_col4:
    fig_amount = px.box(
        results,
        x="Prediction_Label",
        y="Amount",
        color="Prediction_Label",
        color_discrete_map={
            "Genuine": "#10B981",
            "Fraud": "#EF4444"
        },
        title="Transaction Amount by Prediction"
    )

    fig_amount.update_layout(
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827"),
        showlegend=False
    )

    st.plotly_chart(fig_amount, use_container_width=True)


# ==============================
# High Risk Transactions
# ==============================
st.markdown('<div class="section-title">🚨 Top High-Risk Transactions</div>', unsafe_allow_html=True)

high_risk = results.sort_values(
    by="Fraud_Probability",
    ascending=False
).head(10)

st.dataframe(
    high_risk,
    use_container_width=True,
    height=350
)


# ==============================
# Full Results
# ==============================
st.markdown('<div class="section-title">📋 Complete Prediction Results</div>', unsafe_allow_html=True)

display_columns = ["Fraud_Probability", "Prediction_Label"] + [
    col for col in results.columns
    if col not in ["Fraud_Probability", "Prediction_Label", "Prediction"]
]

st.dataframe(
    results[display_columns],
    use_container_width=True,
    height=450
)


# ==============================
# Download
# ==============================
csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Prediction Results",
    data=csv,
    file_name="fraud_detection_predictions.csv",
    mime="text/csv"
)


# ==============================
# Footer
# ==============================
st.markdown("""
<div class="card">
    <h3>🧠 Model Note</h3>
    <p style="color:#6B7280;">
    This dashboard is designed as a decision-support system. In a real banking environment,
    transactions flagged as suspicious should be reviewed by fraud analysts before final action.
    </p>
</div>
""", unsafe_allow_html=True)