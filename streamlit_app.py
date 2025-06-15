import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

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

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 3.  RISK CATEGORY
# ──────────────────────────────────────────────────────────────
def label_risk(p):
    if p < 0.30:
        return "🟢 Low"
    elif p < 0.60:
        return "🟡 Moderate"
    else:
        return "🔴 High"

# ──────────────────────────────────────────────────────────────
# 4.  HEADER / GUIDE / CSV UPLOAD
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar or **upload a CSV** below.  
        2. See **risk prediction, probability & SHAP explanations**.  
        3. Use **Interactive Feature Impact** dropdown to inspect any feature.  
        4. Download or clear **prediction history** at any time.  
        5. *New:* Try **Use Sample Data** for a quick demo or **Reset Form** to start fresh.
        """
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ──────────────────────────────────────────────────────────────
# 5.  SIDEBAR INPUTS (WITH UNIQUE KEYS)
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        base  = col.split("_")[0]
        tip   = tooltips.get(base, "")
        key   = f"inp_{col}"          # unique key for reset & sample
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(
                col, schema[col].unique(), key=key, help=tip
            )
        else:
            if col in X_stats:
                cmin, cmax, cmean = map(float, [X_stats[col]["min"],
                                                 X_stats[col]["max"],
                                                 X_stats[col]["mean"]])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = st.sidebar.slider(
                col, cmin, cmax, cmean, key=key, help=tip
            )
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 6.  FEATURE 12 – USE SAMPLE DATA & FEATURE 11 – RESET FORM
# ──────────────────────────────────────────────────────────────
sample_employee = {
    "Age": 32,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 1100,
    "Department": "Research & Development",
    "DistanceFromHome": 8,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 2,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5200,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "OverTime": "Yes",
    "PercentSalaryHike": 13,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
}

def load_sample():
    """
    Populate session_state with sample values while guaranteeing that
    numeric values are clamped to the slider range.  No schema lookup,
    so no KeyError.
    """
    for col, val in sample_employee.items():
        key = f"inp_{col}"

        # Treat numeric sample values (int, float) as numeric
        if isinstance(val, (int, float, np.number)):
            if col in X_stats:
                cmin = float(X_stats[col]["min"])
                cmax = float(X_stats[col]["max"])
                val = max(min(val, cmax), cmin)   # clamp
        # For categorical just keep original value
        st.session_state[key] = val 

def reset_form():
    for col in schema.columns:
        st.session_state.pop(f"inp_{col}", None)

use_sample = st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
reset_btn  = st.sidebar.button("🔄 Reset Form", on_click=reset_form)

# ──────────────────────────────────────────────────────────────
# 7.  GET DATAFRAME (sidebar or csv)
# ──────────────────────────────────────────────────────────────
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# ──────────────────────────────────────────────────────────────
# 8.  PREDICTION PIPELINE
# ──────────────────────────────────────────────────────────────
X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_user = X_enc.iloc[[0]]

pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk_cat = label_risk(prob)

# ──────────────────────────────────────────────────────────────
# 9.  METRICS
# ──────────────────────────────────────────────────────────────
st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk_cat)

# ──────────────────────────────────────────────────────────────
# 10.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")
shap_vals = explainer.shap_values(X_user)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig1); plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
fig2, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(fig2); plt.clf()

# Local force or waterfall fallback
st.markdown("### 🎯 Local Force Plot")
try:
    fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0],
                            matplotlib=True, show=False)
    st.pyplot(fig3)
except Exception:
    st.info("Force plot fallback to waterfall.")
    fig3, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=shap_vals[0],
                         base_values=explainer.expected_value,
                         data=X_user.iloc[0]),
        max_display=15, show=False)
    st.pyplot(fig3)
st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")

# ──────────────────────────────────────────────────────────────
# 11.  BATCH SUMMARY (remain unchanged)
# ──────────────────────────────────────────────────────────────
if batch_mode:
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    out = raw_df.copy()
    out["Prediction"]  = np.where(preds == 1, "Yes", "No")
    out["Probability"] = (probs * 100).round(1).astype(str)+" %"
    out["Risk Category"] = [label_risk(p) for p in probs]
    st.markdown("### 📑 Batch Prediction Summary")
    st.dataframe(out)

# ──────────────────────────────────────────────────────────────
# 12.  ADD TO HISTORY
# ──────────────────────────────────────────────────────────────
append_df = raw_df.iloc[[0]].copy()
append_df["Prediction"]  = "Yes" if pred else "No"
append_df["Probability"] = f"{prob:.1%}"
append_df["Risk Category"] = risk_cat
append_df["Timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat(
    [st.session_state["history"], append_df], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv, "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
