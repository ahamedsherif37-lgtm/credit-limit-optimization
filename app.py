
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
 
# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Limit Optimization",
    page_icon="💳",
    layout="wide",
)
 
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
 
MODEL_PATH = os.path.join(MODEL_DIR, "credit_risk_xgboost_pipeline.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "risk_threshold.pkl")
RULES_PATH = os.path.join(MODEL_DIR, "credit_limit_rules.pkl")
 
# Feature order must match training: X = df_ml.drop(columns=['default.payment.next.month'])
FEATURE_ORDER = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "Total_delay_Months", "max_delay", "Payment_Ratio", "Credit_Utilization",
]
 
 
# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model pipeline, decision threshold, and credit-limit rules."""
    missing = [p for p in [MODEL_PATH, THRESHOLD_PATH, RULES_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing
 
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    rules = joblib.load(RULES_PATH)
    return model, threshold, rules, []
 
 
# ----------------------------------------------------------------------
# Core logic (mirrors the notebook exactly)
# ----------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    """Recreate Total_delay_Months, max_delay, Payment_Ratio, Credit_Utilization."""
    row = dict(raw)
 
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    row["Total_delay_Months"] = sum(1 for c in pay_cols if row[c] > 0)
    row["max_delay"] = max(row[c] for c in pay_cols)
    row["Payment_Ratio"] = row["PAY_AMT1"] / (abs(row["BILL_AMT1"]) + 1)
    row["Credit_Utilization"] = row["BILL_AMT1"] / (row["LIMIT_BAL"] + 1)
 
    df = pd.DataFrame([row])
    return df[FEATURE_ORDER]
 
 
def assign_risk(probability: float) -> str:
    if probability >= 0.50:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"
 
 
def recommend_credit_limit(current_limit: float, probability: float, rules: dict | None) -> tuple[float, float]:
    """Apply the same tiered-factor rule used in the notebook. Returns (recommended_limit, factor)."""
    if rules:
        if probability < rules["very_low_risk"]["max_probability"]:
            factor = rules["very_low_risk"]["factor"]
        elif probability < rules["low_risk"]["max_probability"]:
            factor = rules["low_risk"]["factor"]
        elif probability < rules["medium_risk"]["max_probability"]:
            factor = rules["medium_risk"]["factor"]
        else:
            factor = rules["high_risk"]["factor"]
    else:
        if probability < 0.10:
            factor = 1.20
        elif probability < 0.25:
            factor = 1.05
        elif probability < 0.50:
            factor = 0.80
        else:
            factor = 0.60
 
    return round(current_limit * factor, -3), factor
 
 
def risk_color(category: str) -> str:
    return {"Low Risk": "#02C39A", "Medium Risk": "#F9A825", "High Risk": "#D64550"}[category]
 
 
# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("💳 Credit Limit Optimization Using a Risk Model")
st.caption(
    "Enter a customer's profile and repayment history to predict default risk "
    "and get an optimized credit-limit recommendation, powered by a tuned XGBoost model."
)
 
model, threshold, rules, missing_files = load_artifacts()
 
if missing_files:
    st.warning(
        "⚠️ Model files not found in the app folder. Copy these files (created at the end "
        "of the notebook) next to `app.py`, then refresh:\n\n"
        + "\n".join(f"- `{os.path.basename(p)}`" for p in missing_files)
    )
 
st.divider()
 
# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer Details")
 
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        limit_bal = st.number_input("Current Credit Limit (LIMIT_BAL)", min_value=1000, max_value=1_000_000, value=200000, step=1000)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
    with c2:
        sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])
    with c3:
        marriage = st.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
    with c4:
        st.markdown("&nbsp;")
 
    st.markdown("**Repayment Status (last 6 months)** — -2/-1/0 = paid on time, 1+ = months delayed")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1: pay_0 = st.number_input("PAY_0 (recent)", min_value=-2, max_value=8, value=0)
    with p2: pay_2 = st.number_input("PAY_2", min_value=-2, max_value=8, value=0)
    with p3: pay_3 = st.number_input("PAY_3", min_value=-2, max_value=8, value=0)
    with p4: pay_4 = st.number_input("PAY_4", min_value=-2, max_value=8, value=0)
    with p5: pay_5 = st.number_input("PAY_5", min_value=-2, max_value=8, value=0)
    with p6: pay_6 = st.number_input("PAY_6", min_value=-2, max_value=8, value=0)
 
    st.markdown("**Bill Amounts (last 6 months)**")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    with b1: bill1 = st.number_input("BILL_AMT1", value=50000, step=1000)
    with b2: bill2 = st.number_input("BILL_AMT2", value=48000, step=1000)
    with b3: bill3 = st.number_input("BILL_AMT3", value=46000, step=1000)
    with b4: bill4 = st.number_input("BILL_AMT4", value=44000, step=1000)
    with b5: bill5 = st.number_input("BILL_AMT5", value=42000, step=1000)
    with b6: bill6 = st.number_input("BILL_AMT6", value=40000, step=1000)
 
    st.markdown("**Payment Amounts (last 6 months)**")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1: pay_amt1 = st.number_input("PAY_AMT1", min_value=0, value=2000, step=500)
    with a2: pay_amt2 = st.number_input("PAY_AMT2", min_value=0, value=2000, step=500)
    with a3: pay_amt3 = st.number_input("PAY_AMT3", min_value=0, value=2000, step=500)
    with a4: pay_amt4 = st.number_input("PAY_AMT4", min_value=0, value=2000, step=500)
    with a5: pay_amt5 = st.number_input("PAY_AMT5", min_value=0, value=2000, step=500)
    with a6: pay_amt6 = st.number_input("PAY_AMT6", min_value=0, value=2000, step=500)
 
    submitted = st.form_submit_button("🔍 Predict Risk & Recommend Limit", use_container_width=True)
 
# ----------------------------------------------------------------------
# Prediction & results
# ----------------------------------------------------------------------
if submitted:
    if model is None:
        st.error("Cannot run a prediction — the model files are missing. See the warning above.")
    else:
        raw_input = {
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
            "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
            "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
        }
 
        X_input = engineer_features(raw_input)
        probability = float(model.predict_proba(X_input)[:, 1][0])
        risk_category = assign_risk(probability)
        recommended_limit, factor = recommend_credit_limit(limit_bal, probability, rules)
 
        st.divider()
        st.subheader("Prediction Results")
 
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Probability", f"{probability:.1%}")
        r2.metric("Risk Category", risk_category)
        r3.metric("Current Limit", f"${limit_bal:,.0f}")
        delta = recommended_limit - limit_bal
        r4.metric("Recommended Limit", f"${recommended_limit:,.0f}", delta=f"{delta:+,.0f}")
 
        col_left, col_right = st.columns([1, 1])
 
        with col_left:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Default Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color(risk_category)},
                        "steps": [
                            {"range": [0, 30], "color": "#E8F8F3"},
                            {"range": [30, 50], "color": "#FFF3D6"},
                            {"range": [50, 100], "color": "#FBE4E6"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": (threshold or 0.30) * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
 
        with col_right:
            comp_fig = go.Figure(
                data=[
                    go.Bar(name="Current Limit", x=["Credit Limit"], y=[limit_bal], marker_color="#1E2761"),
                    go.Bar(name="Recommended Limit", x=["Credit Limit"], y=[recommended_limit], marker_color=risk_color(risk_category)),
                ]
            )
            comp_fig.update_layout(
                barmode="group", height=320, title="Current vs Recommended Limit",
                margin=dict(l=20, r=20, t=50, b=10), yaxis_title="Amount ($)",
            )
            st.plotly_chart(comp_fig, use_container_width=True)
 
        st.info(
            f"**Interpretation:** This customer has a **{probability:.1%}** estimated probability of "
            f"default, placing them in the **{risk_category}** category (decision threshold = "
            f"{(threshold or 0.30):.2f}). Based on the recommendation rule (factor = **{factor}x**), "
            f"the suggested credit limit is **${recommended_limit:,.0f}** "
            f"({'an increase' if delta > 0 else 'a decrease' if delta < 0 else 'no change'} of "
            f"${abs(delta):,.0f} from the current limit)."
        )
 
        with st.expander("View engineered feature values sent to the model"):
            st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)
 
st.divider()
st.caption(
    "Model: Tuned XGBoost classifier (ROC-AUC ≈ 0.78) · Features: 27 (23 raw + 4 engineered) · "
    "Decision threshold optimized for F1-score · Built with Streamlit"
)
 






