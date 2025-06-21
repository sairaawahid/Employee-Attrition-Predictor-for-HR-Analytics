import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# ==============================
# 1.  Load Cached Resources
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    return json.loads(Path("employee_schema.json").read_text())

@st.cache_data
def load_tooltips():
    try:
        return json.loads(Path("feature_tooltips.json").read_text())
    except Exception:
        return {}

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ==============================
# 2.  Init session state & Clear History logic
# ==============================
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()
if "just_cleared_history" not in st.session_state:
    st.session_state["just_cleared_history"] = False

# --- Clear History button ---
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.session_state["just_cleared_history"] = True
    st.rerun()

if st.session_state.get("just_cleared_history", False):
    st.session_state["just_cleared_history"] = False
    st.stop()

# ==============================
# 3.  Load Model and Data
# ==============================
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

# ==============================
# 4.  Helpers
# ==============================
def label_risk(p):
    if p < .30:
        return "🟢 Low"
    if p < .60:
        return "🟡 Moderate"
    return "🔴 High"

def safe_stats(col):
    meta  = schema_meta.get(col, {})
    cmin  = meta.get("min", 0.0)
    cmax  = meta.get("max", 1.0)
    cmean = meta.get("mean", (cmin + cmax) / 2)
    if cmin == cmax:
        cmax = cmin + 1
    return float(cmin), float(cmax), float(cmean)

# ==============================
# 5.  UI Header & Guide
# ==============================
st.title("Employee Attrition Predictor")
st.markdown(
    "A decision-support tool for HR professionals to predict employee attrition and understand the key reasons behind the prediction. "
    "Get clear insights with probability scores, risk levels, and SHAP-powered visual explanations for informed talent management."
)
with st.expander("**How to use this app**", expanded=False):
    st.markdown(
        """
1. **Enter employee details** in the sidebar or **Use Sample Data** for a demo.
2. Click **Reset Form** to start fresh.
3. **Upload a CSV (optional)** for bulk scoring and row-by-row inspection.  
4. Click **Run Prediction** to see risk, probability & risk category.  
5. Explore **SHAP plots** to understand which factors drive each prediction.  
6. Use the **Interactive Feature Impact** to inspect any feature.  
7. **Download or Clear History** to track past predictions and share insights.
        """
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ==============================
# 6.  Sidebar widgets
# ==============================
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
    row = {}
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")
        if meta["dtype"] == "object":
            options = meta.get("options", ["Unknown"])
            cur = st.session_state.get(key, options[0])
            if cur not in options:
                cur = options[0]
            row[col] = st.sidebar.selectbox(col, options, key=key, help=tip)
        else:
            cmin, cmax, _ = safe_stats(col)
            cur = float(st.session_state.get(key, cmin))
            cur = min(max(cur, cmin), cmax)
            discrete = meta.get("discrete", False)
            step = 1.0 if discrete else 0.1
            if abs(cmax - cmin) < 1e-9:
                row[col] = st.sidebar.number_input(col, value=cur, key=key, help=tip)
            else:
                row[col] = st.sidebar.slider(col, cmin, cmax, step=step, key=key, help=tip)
    return pd.DataFrame([row])

sample_employee = {
    "Age": 32,
    "Attrition": "No",
    "Business Travel": "Travel_Rarely",
    "Daily Rate": 1100,
    "Department": "Research & Development",
    "Distance From Home": 8,
    "Education": "Bachelor's",
    "Education Field": "Life Sciences",
    "Environment Satisfaction": 3,
    "Gender": "Male",
    "Hourly Rate": 65,
    "Job Involvement": 3,
    "Job Level": 2,
    "Job Role": "Research Scientist",
    "Job Satisfaction": 2,
    "Marital Status": "Single",
    "Monthly Income": "5 000 – 5 999",
    "Monthly Rate": "10 000 – 14 999",
    "No. of Companies Worked": 2,
    "Over Time": "Yes",
    "Percent Salary Hike": 13,
    "Performance Rating": 3,
    "Relationship Satisfaction": 2,
    "Stock Option Level": 1,
    "Total Working Years": 10,
    "Training Times Last Year": 3,
    "Work Life Balance": 2,
    "Years At Company": 5,
    "Years In Current Role": 3,
    "Years Since Last Promotion": 1,
    "Years With Current Manager": 2,
}

def load_sample():
    for col, val in sample_employee.items():
        if col not in schema_meta:
            continue
        if schema_meta[col]["dtype"] != "object":
            cmin, cmax, _ = safe_stats(col)
            val = max(min(val, cmax), cmin)
        st.session_state[f"inp_{col}"] = val

def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        if meta["dtype"] == "object":
            st.session_state[key] = meta.get("options", ["Unknown"])[0]
        else:
            st.session_state[key] = safe_stats(col)[0]

st.sidebar.button("Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form", on_click=reset_form)

# ==============================
# 7.  Data Collection & Run Prediction
# ==============================
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

run_pred = st.sidebar.button("Run Prediction", use_container_width=True)

# If batch_mode, support row selection (instant, no Run Prediction click needed)
if batch_mode and run_pred:
    # Add "Row" column starting from 1
    raw_df_display = raw_df.copy()
    raw_df_display.insert(0, "Row", np.arange(1, len(raw_df) + 1))
    # Run predictions for all
    X_full = pd.concat([raw_df, pd.DataFrame([{c: (m["options"][0] if m["dtype"] == "object" else 0)
                                              for c, m in schema_meta.items()}])], ignore_index=True)
    X_enc = pd.get_dummies(X_full).iloc[:len(raw_df)]
    X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    raw_df_display["Prediction"]    = np.where(preds == 1, "Yes", "No")
    raw_df_display["Probability"]   = (probs * 100).round(1).astype(str) + " %"
    raw_df_display["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("Batch Prediction Summary")
    st.info(
        "This table summarizes attrition predictions for all uploaded employees. "
        "Each row shows whether the employee is predicted to leave (Yes/No), "
        "the exact probability, and the assigned risk category: "
        "**Low (<30%)**, **Moderate (30–60%)**, or **High (>60%)**. "
        "Select a row for detailed SHAP analysis."
    )
    st.dataframe(raw_df_display, use_container_width=True)

    row_labels = [f"Row {i}" for i in range(1, len(raw_df)+1)]
    default_idx = 0
    sel_label = st.selectbox("Select row for SHAP explanation", row_labels, index=default_idx)
    sel_row = int(sel_label.split(" ")[-1]) - 1

    X_user  = X_enc.iloc[[sel_row]]
    user_df = raw_df_display.iloc[[sel_row]]

    # -- show results instantly, don't require Run Prediction again --
    pred = preds[sel_row]
    prob = probs[sel_row]
    risk = label_risk(prob)
else:
    # Only proceed if run_pred is pressed (not batch or not uploaded)
    if not run_pred:
        st.stop()

    template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
                for c, m in schema_meta.items()}
    schema_df = pd.DataFrame([template])
    X_full = pd.concat([raw_df, schema_df], ignore_index=True)
    X_enc  = pd.get_dummies(X_full).iloc[: len(raw_df)]
    X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)
    X_user  = X_enc.iloc[[0]]
    user_df = raw_df.iloc[[0]]

    pred = model.predict(X_user)[0]
    prob = model.predict_proba(X_user)[0, 1]
    risk = label_risk(prob)

# ==============================
# 8.  Prediction display & SHAP
# ==============================
st.markdown("### Prediction Results")
st.info(
    "Below you’ll see whether the employee is likely to leave the company (Yes/No), "
    "the exact probability, and the calibrated risk category."
    "**Low (<30%)**, **Moderate (30–60%)**, or **High (>60%)**."
)
st.markdown(
    f"""
<div style='border:2px solid #eee;border-radius:10px;padding:20px;background:#f9f9f9;'>
  <div style='display:flex;justify-content:space-between;font-size:18px;'>
    <div><strong>Prediction</strong><br>
         <span style='font-size:24px;color:#444;'>{'Yes' if pred else 'No'}</span></div>
    <div><strong>Probability</strong><br>
         <span style='font-size:24px;color:#444;'>{prob:.1%}</span></div>
    <div><strong>Risk Category</strong><br>
         <span style='font-size:24px;'>{risk}</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("🔍 SHAP Explanations")
st.info(
    "These plots show **which features push the prediction higher or lower.** "
    "▲ Positive SHAP pushes toward leaving; ▼ Negative pushes toward staying."
)
sv = explainer.shap_values(X_user)
if isinstance(sv, list):
    sv = sv[1]

st.markdown("### Global Impact — Beeswarm")
fig_bsw, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig_bsw); plt.clf()

st.markdown("### Decision Path")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(fig_dec); plt.clf()

st.markdown("### Local Force Plot")
try:
    fig_f = shap.plots.force(
        explainer.expected_value,
        sv[0],
        X_user.iloc[0],
        matplotlib=True,
        show=False,
    )
    st.pyplot(fig_f)
except Exception:
    fig_f, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=X_user.iloc[0],
        ),
        max_display=15,
        show=False,
    )
    st.pyplot(fig_f)

st.markdown("### Interactive Feature Impact")
feature = st.selectbox("Choose feature", X_user.columns, key="feat_sel")
idx = X_user.columns.get_loc(feature)
val = sv[0][idx] if sv.ndim == 2 else sv[idx]
fig_bar, _ = plt.subplots()
shap.bar_plot(np.array([val]), feature_names=[feature], max_display=1, show=False)
st.pyplot(fig_bar); plt.clf()

# ==============================
# 9.  History (append & display)
# ==============================
# Only append if NOT just cleared and only for single prediction or for selected row in batch mode
if not st.session_state.get("just_cleared_history", False):
    df_to_append = user_df.copy()
    df_to_append["Prediction"] = "Yes" if pred else "No"
    df_to_append["Probability"] = f"{prob:.1%}"
    df_to_append["Risk Category"] = risk
    df_to_append["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["history"] = pd.concat(
        [st.session_state["history"], df_to_append], ignore_index=True
    )

st.subheader("Prediction History")
st.dataframe(st.session_state["history"], use_container_width=True)
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button(
    "💾 Download History", csv_hist, "prediction_history.csv", "text/csv"
)
