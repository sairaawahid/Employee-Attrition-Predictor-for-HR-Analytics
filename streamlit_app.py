import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1.  CACHED LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    return pd.read_json("employee_schema.json")

@st.cache_data
def load_stats():
    return json.loads(Path("numeric_stats.json").read_text())

@st.cache_data
def load_tooltips():
    return json.loads(Path("feature_tooltips.json").read_text())

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model      = load_model()
schema     = load_schema()
X_stats    = load_stats()
tooltips   = load_tooltips()
explainer  = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  HEADER & GUIDE
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar *or* **upload a CSV** below.  
        2. View predicted **attrition risk, probability and risk card**.  
        3. Use **SHAP charts** and **row selector** (for CSV) to inspect individuals.  
        4. Apply the **psychology-based tips** to design HR interventions.
        """
    )

# 🔽 CSV uploader now lives right here (not in sidebar)
uploaded = st.file_uploader("📂 Upload Employee CSV (optional)", type="csv")

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR FORM (single employee)
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

def user_input_features():
    data = {}
    for col in schema.columns:
        base = col.split("_")[0]
        tip  = tooltips.get(base, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique(), help=tip)
        else:
            if col in X_stats:
                cmin, cmax, cmean = map(float, [X_stats[col]["min"], X_stats[col]["max"], X_stats[col]["mean"]])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = st.sidebar.slider(col, cmin, cmax, cmean, help=tip)
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 5.  BUILD INPUT DF (single or batch)
# ──────────────────────────────────────────────────────────────
if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.success(f"Loaded **{len(raw_df)}** employees from CSV.")
else:
    raw_df = user_input_features()

# One-hot encode to training schema
X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_pred = X_enc.iloc[: len(raw_df)]

# ──────────────────────────────────────────────────────────────
# 6.  PREDICTION & DISPLAY
# ──────────────────────────────────────────────────────────────
preds = model.predict(X_pred)
probs = model.predict_proba(X_pred)[:, 1]

if len(raw_df) > 1:                   # batch mode
    results = raw_df.copy()
    results["Prediction"]  = np.where(preds == 1, "Yes", "No")
    results["Probability"] = (probs * 100).round(1).astype(str) + " %"
    results["Risk"]        = pd.cut(probs, [0, .3, .6, 1], labels=["Low", "Moderate", "High"])
    st.subheader("📑 Batch Predictions")
    st.dataframe(results)

    row_idx = st.number_input("Select employee row to inspect:", 0, len(raw_df)-1, 0)
    input_df = raw_df.iloc[[row_idx]]
    X_user   = X_pred.iloc[[row_idx]]
    pred, prob = preds[row_idx], probs[row_idx]
else:                                 # single mode
    input_df = raw_df
    X_user   = X_pred
    pred, prob = preds[0], probs[0]

# ──────────────────────────────────────────────────────────────
# 7.  METRIC CARDS
# ──────────────────────────────────────────────────────────────
st.subheader("Prediction")

risk_tag = "🟢 Low" if prob < .3 else "🟡 Moderate" if prob < .6 else "🔴 High"
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk_tag)

# ──────────────────────────────────────────────────────────────
# 8.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

shap_vals = explainer.shap_values(X_user)
shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig1); plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
fig2, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(fig2); plt.clf()

st.markdown("### 🎯 Local Force Plot")
fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0],
                        matplotlib=True, show=False)
st.pyplot(fig3)
st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")

# ──────────────────────────────────────────────────────────────
# 9.  PSYCHOLOGY-BASED HR RECOMMENDATIONS (KeyError-safe)
# ──────────────────────────────────────────────────────────────
st.subheader("🧠 Psychology-Based HR Recommendations")

rec_map = {
    "JobSatisfaction": {
        1: "Very low job satisfaction – explore role fit or engagement programs.",
        2: "Moderate dissatisfaction – mentoring or job enrichment may help.",
        3: "Generally satisfied – maintain engagement.",
        4: "Highly satisfied – continue supporting growth."
    },
    "EnvironmentSatisfaction": {
        1: "Poor environment rating – review ergonomics and team climate.",
        2: "Mediocre rating – gather feedback for improvements.",
        3: "Supportive environment.",
        4: "Excellent environment satisfaction."
    },
    "RelationshipSatisfaction": {
        1: "Poor coworker relations – consider team-building or mediation.",
        2: "Average relations – encourage open communication.",
        3: "Healthy relations.",
        4: "Strong coworker relationships."
    },
    "JobInvolvement": {
        1: "Low involvement – clarify goals and recognize achievements.",
        2: "Could benefit from intrinsic motivators.",
        3: "Good engagement.",
        4: "Highly involved – potential leader."
    },
    "WorkLifeBalance": {
        1: "Poor balance – offer flexibility or workload review.",
        2: "Risk of imbalance – monitor hours.",
        3: "Healthy balance.",
        4: "Excellent balance."
    },
    "OverTime_Yes": "Regular overtime detected – assess workload and risk of burnout."
}

tips = []

def safe_value(df, col):
    return df[col].values[0] if col in df.columns else None

for col in ["JobSatisfaction", "EnvironmentSatisfaction",
            "RelationshipSatisfaction", "JobInvolvement",
            "WorkLifeBalance"]:
    val = safe_value(input_df, col)
    if val in rec_map[col]:
        tips.append(rec_map[col][val])

# Overtime flag (needs one-hot)
if "OverTime_Yes" in X_user.columns and X_user["OverTime_Yes"].iloc[0] == 1:
    tips.append(rec_map["OverTime_Yes"])

if tips:
    for t in tips:
        st.info(t)
else:
    st.success("No major psychological red flags detected; continue current support.")
