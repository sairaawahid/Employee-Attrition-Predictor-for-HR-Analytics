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
def load_schema_json():
    """Load the corrected employee_schema.json (dtype + options)."""
    return json.loads(Path("employee_schema.json").read_text())

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
model       = load_model()
schema_meta = load_schema_json()      # dict: {col: {"dtype": "...", ...}}
X_stats     = load_stats()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 3.  RISK CATEGORY HELPER
# ──────────────────────────────────────────────────────────────
def label_risk(p):
    if p < 0.30:   return "🟢 Low"
    if p < 0.60:   return "🟡 Moderate"
    return "🔴 High"

# ──────────────────────────────────────────────────────────────
# 4.  PAGE HEADER / CSV UPLOAD
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar or **upload a CSV** below.  
        2. View **prediction, probability, risk category & SHAP explanations**.  
        3. **Interactive Feature Impact** lets you inspect any feature.  
        4. Download / clear **prediction history** anytime.  
        5. Click **Use Sample Data** for a quick demo or **Reset Form** to start fresh.
        """
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ──────────────────────────────────────────────────────────────
# 5.  SIDEBAR WIDGETS
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
    data = {}
    for col, meta in schema_meta.items():
        tip = tooltips.get(col.split("_")[0], "")
        key = f"inp_{col}"

        # Categorical dropdown
        if meta["dtype"] == "object":
            options = meta["options"]
            default = st.session_state.get(key, options[0])
            if default not in options:
                default = options[0]
            data[col] = st.sidebar.selectbox(
                col, options,
                index=options.index(default),
                key=key, help=tip
            )
        # Numeric slider
        else:
            cmin, cmax = X_stats[col]["min"], X_stats[col]["max"]
            cmean      = X_stats[col]["mean"]
            default    = float(st.session_state.get(key, cmean))
            default    = max(min(default, cmax), cmin)
            data[col]  = st.sidebar.slider(
                col, cmin, cmax, default,
                key=key, help=tip
            )
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 6.  SAMPLE DATA & RESET FORM
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
    for col, val in sample_employee.items():
        key = f"inp_{col}"
        if isinstance(val, (int, float, np.number)) and col in X_stats:
            cmin, cmax = X_stats[col]["min"], X_stats[col]["max"]
            val = max(min(val, cmax), cmin)
        st.session_state[key] = val

def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        if meta["dtype"] == "object":
            st.session_state[key] = meta["options"][0]
        else:
            st.session_state[key] = X_stats[col]["mean"]

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form",     on_click=reset_form)

# ──────────────────────────────────────────────────────────────
# 7.  GET DATAFRAME (sidebar or csv)
# ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# ──────────────────────────────────────────────────────────────
# 8.  PREDICTION PIPELINE
# ──────────────────────────────────────────────────────────────
# Build a template row to guarantee all categories appear after get_dummies
template_row = {
    col: (meta["options"][0] if meta["dtype"] == "object" else 0)
    for col, meta in schema_meta.items()
}
schema_df = pd.DataFrame([template_row])

X_full = pd.concat([raw_df, schema_df], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[0:len(raw_df)]   # keep only user rows
X_user = X_enc.iloc[[0]]

pred      = model.predict(X_user)[0]
prob      = model.predict_proba(X_user)[0, 1]
risk_cat  = label_risk(prob)

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

st.markdown("### 🎯 Local Force Plot")
try:
    fig3 = shap.plots.force(
        explainer.expected_value, shap_vals[0], X_user.iloc[0],
        matplotlib=True, show=False
    )
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
# 11.  BATCH SUMMARY
# ──────────────────────────────────────────────────────────────
if batch_mode:
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    out = raw_df.copy()
    out["Prediction"]    = np.where(preds == 1, "Yes", "No")
    out["Probability"]   = (probs * 100).round(1).astype(str) + " %"
    out["Risk Category"] = [label_risk(p) for p in probs]
    st.markdown("### 📑 Batch Prediction Summary")
    st.dataframe(out)

# ──────────────────────────────────────────────────────────────
# 12.  HISTORY
# ──────────────────────────────────────────────────────────────
append_df = raw_df.iloc[[0]].copy()
append_df["Prediction"]   = "Yes" if pred else "No"
append_df["Probability"]  = f"{prob:.1%}"
append_df["Risk Category"] = risk_cat
append_df["Timestamp"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat(
    [st.session_state["history"], append_df], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
