# ===== PATCH FOR joblib.Bunch ERROR =====
import sys
try:
    from sklearn.utils import Bunch
    sys.modules['joblib.Bunch'] = Bunch
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════
# 1.  Load Cached Resources
# ═══════════════════════════════════════
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

# ═══════════════════════════════════════
# 2.  Initialise session state and model
# ═══════════════════════════════════════
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
if "just_cleared_history" not in st.session_state:
    st.session_state["just_cleared_history"] = False

model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

# ═══════════════════════════════════════
# 3.  Clear History Logic
# ═══════════════════════════════════════
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.session_state["prediction_done"] = False
    st.session_state["just_cleared_history"] = True
    st.experimental_rerun()

if st.session_state.get("just_cleared_history", False):
    st.session_state["just_cleared_history"] = False
    st.stop()

# ═══════════════════════════════════════
# 4.  UI Header & Guide
# ═══════════════════════════════════════
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
        """)

# ═══════════════════════════════════════
# 5.  Sidebar widgets
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
    row = {}
    for col, meta in schema_meta.items():
        key, tip = f"inp_{col}", tooltips.get(col.split("_")[0], "")
        if meta["dtype"] == "object":
            opts = meta["options"]
            row[col] = st.sidebar.selectbox(
                col, opts,
                index=opts.index(st.session_state.get(key, opts[0])),
                key=key, help=tip
            )
        else:
            cmin, cmax, _ = safe_stats(col)
            cur = float(st.session_state.get(key, cmin))
            cur = min(max(cur, cmin), cmax)
            step = 1.0 if meta.get("discrete", False) else 0.1
            row[col] = st.sidebar.slider(col, cmin, cmax, value=cur, step=step, key=key, help=tip)
    return pd.DataFrame([row])

def label_risk(p):
    return "🟢 Low" if p < .30 else "🟡 Moderate" if p < .60 else "🔴 High"

def safe_stats(col):
    meta  = schema_meta.get(col, {})
    cmin  = meta.get("min", 0)
    cmax  = meta.get("max", 1)
    cmean = meta.get("mean", (cmin + cmax) / 2)
    if cmin == cmax:
        cmax = cmin + 1   
    return float(cmin), float(cmax), float(cmean)

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
    for c, v in sample_employee.items():
        st.session_state[f"inp_{c}"] = v

def reset_form():
    for c, meta in schema_meta.items():
        st.session_state[f"inp_{c}"] = meta["options"][0] if meta["dtype"] == "object" else safe_stats(c)[0]

st.sidebar.button("Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form", on_click=reset_form)

# ═══════════════════════════════════════
# 6.  Collect data *without* running model
# ═══════════════════════════════════════
if (file := st.file_uploader("📂 Upload CSV (optional)", type="csv")):
    raw_df, batch_mode = pd.read_csv(file), True
else:
    raw_df, batch_mode = sidebar_inputs(), False

run_pred = st.sidebar.button("Run Prediction", use_container_width=True)

if not run_pred:
    st.session_state["prediction_done"] = False
    st.stop()

# ═══════════════════════════════════════
# 7.  Prepare data for model (after click)
# ═══════════════════════════════════════
template = {c: (meta["options"][0] if meta["dtype"] == "object" else 0)
            for c, meta in schema_meta.items()}
X_full = pd.concat([raw_df, pd.DataFrame([template])], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

# ═══════════════════════════════════════
# 8.  Prediction logic
# ═══════════════════════════════════════
if batch_mode:
    preds  = model.predict(X_enc)
    probs  = model.predict_proba(X_enc)[:, 1]
    raw_df["Prediction"]    = np.where(preds == 1, "Yes", "No")
    raw_df["Probability"]   = (probs * 100).round(1).astype(str) + " %"
    raw_df["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("Batch Prediction Summary")
    st.info(
        "This table summarizes attrition predictions for all uploaded employees. "
        "Each row shows whether the employee is predicted to leave (Yes/No), "
        "the exact probability, and the assigned risk category: "
        "**Low (<30%)**, **Moderate (30–60%)**, or **High (>60%)**. "
        "Select a row for detailed SHAP analysis."
    )

    sel_row = st.number_input(
        "Row to inspect (1-based)",
        min_value=1,
        max_value=len(raw_df),
        value=1,
        key="row_picker",
    )
    st.dataframe(raw_df, use_container_width=True)
    X_user = X_enc.iloc[[sel_row - 1]]
    user_df = raw_df.iloc[[sel_row - 1]]
else:
    X_user  = X_enc.iloc[[0]]
    user_df = raw_df.iloc[[0]]

# ═══════════════════════════════════════
# 9.  Single prediction display
# ═══════════════════════════════════════
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk = label_risk(prob)

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

# ═══════════════════════════════════════
# 10.  SHAP explanations
# ═══════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
st.info(
    "These plots show **which features push the prediction higher or lower.** "
    "▲ Positive SHAP pushes toward leaving; ▼ Negative pushes toward staying."
)
sv = explainer.shap_values(X_user)
if isinstance(sv, list):
    sv = sv[1]

st.markdown("### Global Impact — Beeswarm")
st.info("This plot shows which features **had the highest overall impact** "
        "on the model’s prediction for this employee. Longer bars = stronger effect. "
        "Colors indicate whether the value pushed the prediction higher (red) or lower (blue).")
fig_bsw, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig_bsw); plt.clf()

st.markdown("### Decision Path")
st.info("This plot explains the **sequence of contributions** each feature made, "
        "starting from the model’s baseline prediction. Features that increased or "
        "decreased the risk are shown from left to right, helping you follow the model’s logic.")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(fig_dec); plt.clf()

st.markdown("### Local Force Plot")
st.info("This plot provides a **visual tug-of-war**: features pushing the prediction "
        "higher (red) vs. lower (blue). It gives an intuitive sense of what tipped the balance "
        "towards a high or low attrition risk for this specific case.")
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
    st.info("Force plot unavailable – showing waterfall instead.")
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
st.info("Select a feature to see **how much it individually influenced** the prediction. "
        "This bar shows whether the chosen feature increased or decreased attrition risk "
        "and by how much in the context of this specific employee.")
feature = st.selectbox("Choose feature", X_user.columns, key="feat_sel")
idx = X_user.columns.get_loc(feature)
val = sv[0][idx] if sv.ndim == 2 else sv[idx]
fig_bar, _ = plt.subplots()
shap.bar_plot(np.array([val]), feature_names=[feature], max_display=1, show=False)
st.pyplot(fig_bar); plt.clf()

# ═══════════════════════════════════════
# 11.  Append to history *once* per click
# ═══════════════════════════════════════
if not st.session_state["prediction_done"]:
    hd = user_df.copy()
    hd["Prediction"] = "Yes" if pred else "No"
    hd["Probability"] = f"{prob:.1%}"
    hd["Risk Category"] = risk
    hd["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["history"] = pd.concat([st.session_state["history"], hd], ignore_index=True)
st.session_state["prediction_done"] = True

# ═══════════════════════════════════════
# 12.  History display / download
# ═══════════════════════════════════════
st.subheader("Prediction History")
st.dataframe(st.session_state["history"], use_container_width=True)

csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist, "prediction_history.csv", "text/csv")
