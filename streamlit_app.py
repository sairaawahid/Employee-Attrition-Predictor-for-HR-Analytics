import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

st.set_page_config(layout="wide", page_title="Employee Attrition Predictor")

# ──────────────────────────────────────────────────────────────
# 1. LOAD MODEL & METADATA
# ──────────────────────────────────────────────────────────────
model = joblib.load("xgboost_optimized_model.pkl")

with open("employee_schema.json") as f:
    schema = json.load(f)
with open("feature_tooltips.json") as f:
    tooltips = json.load(f)

explainer = shap.TreeExplainer(model)

# ──────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def is_cat(col):       # schema flag
    return schema[col]["dtype"] == "object"

def risk_label(p):
    return "🟢 Low" if p < 0.30 else "🟡 Moderate" if p < 0.60 else "🔴 High"

# sample row (same keys as schema)
sample_employee = {
    "Age": 32, "BusinessTravel": "Travel_Rarely", "DailyRate": 1100,
    "Department": "Research & Development", "DistanceFromHome": 8,
    "Education": 3, "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 65,
    "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Research Scientist",
    "JobSatisfaction": 2, "MaritalStatus": "Single", "MonthlyIncome": 5200,
    "MonthlyRate": 14000, "NumCompaniesWorked": 2, "OverTime": "Yes",
    "PercentSalaryHike": 13, "PerformanceRating": 3,
    "RelationshipSatisfaction": 2, "StockOptionLevel": 1,
    "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 2, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
}

# ──────────────────────────────────────────────────────────────
# 3. SIDEBAR INPUTS
# ──────────────────────────────────────────────────────────────
def sidebar_inputs():
    data = {}
    for col, meta in schema.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")
        if is_cat(col):                                   # categorical
            opts = meta["options"]
            current = st.session_state.get(key, opts[0])
            if current not in opts:
                current = opts[0]
            data[col] = st.sidebar.selectbox(
                col, opts, index=opts.index(current), key=key, help=tip
            )
        else:                                             # numeric
            cmin, cmax, cmean = meta["min"], meta["max"], meta["mean"]
            is_int = str(meta["dtype"]).startswith("int")
            cast   = (lambda x: int(round(x))) if is_int else float
            cmin, cmax, cmean = cast(cmin), cast(cmax), cast(cmean)

            session_val = st.session_state.get(key, cmean)
            try: session_val = cast(session_val)
            except Exception: session_val = cmean
            session_val = max(min(session_val, cmax), cmin)

            # If range collapsed or slider fails, fallback to number_input
            slider_possible = (abs(cmax - cmin) > 1e-9)
            if not slider_possible:
                data[col] = st.sidebar.number_input(col, value=session_val, key=key, help=tip)
            else:
                try:
                    data[col] = st.sidebar.slider(
                        col, cmin, cmax, session_val, key=key, help=tip
                    )
                except Exception:
                    data[col] = st.sidebar.number_input(
                        col, min_value=cmin, max_value=cmax,
                        value=session_val, key=key, help=tip
                    )
    return pd.DataFrame([data])

# callbacks
def reset_form():
    for col, meta in schema.items():
        key = f"inp_{col}"
        st.session_state[key] = meta["options"][0] if is_cat(col) else meta["mean"]

def load_sample():
    for col, val in sample_employee.items():
        st.session_state[f"inp_{col}"] = val

# ──────────────────────────────────────────────────────────────
# 4. HEADER & GUIDE
# ──────────────────────────────────────────────────────────────
st.title("📊 Employee Attrition Predictor")

with st.expander("ℹ️ Guide", expanded=True):
    st.markdown("""
1. **Enter data** in the sidebar or **upload CSV** for batch prediction  
2. Press **Predict** to see risk & SHAP explanations  
3. Use **Sample Data** for demo, **Reset** to clear
""")
    colA, colB = st.columns([1, 4])
    colA.button("🔁 Reset", on_click=reset_form)
    colB.button("🧪 Sample Data", on_click=load_sample)

# ──────────────────────────────────────────────────────────────
# 5. CSV UPLOAD OR SINGLE ENTRY
# ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV (optional)", type="csv")
if uploaded:
    raw_df = pd.read_csv(uploaded)
    batch  = True
else:
    raw_df = sidebar_inputs()
    batch  = False

# ──────────────────────────────────────────────────────────────
# 6. PREDICT BUTTON
# ──────────────────────────────────────────────────────────────
if st.button("🔮 Predict"):
    # one-hot encode, align to model
    X = pd.get_dummies(raw_df)
    for feat in model.feature_names_in_:
        if feat not in X:
            X[feat] = 0
    X = X[model.feature_names_in_]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    out = raw_df.copy()
    out["Prediction"]  = np.where(preds == 1, "Yes", "No")
    out["Probability"] = (probs * 100).round(2)
    out["Risk"]        = [risk_label(p) for p in probs]

    st.subheader("Results")
    st.dataframe(out)

    # SHAP beeswarm
    st.subheader("Global SHAP Impact")
    shap_vals = explainer(X)
    st.pyplot(shap.plots.beeswarm(shap_vals, show=False))

    # Local force / waterfall for first row
    if not batch:
        st.subheader("Local Explanation")
        try:
            fig = shap.plots.force(
                explainer.expected_value[1], shap_vals[0].values,
                X.iloc[0], matplotlib=True, show=False
            )
            st.pyplot(fig)
        except Exception:
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_vals[0], max_display=15, show=False)
            st.pyplot(fig)

    # download
    st.download_button(
        "Download CSV", out.to_csv(index=False), "attrition_predictions.csv", "text/csv"
    )

