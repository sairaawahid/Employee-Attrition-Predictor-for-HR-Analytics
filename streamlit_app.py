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
# 1 . Cached resources
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
# 2 . Session-state keys
# ═══════════════════════════════════════
ss = st.session_state
for k, v in {
        "history"          : pd.DataFrame(),
        "predicted"        : False,   # True after first Run Prediction
        "just_cleared"     : False,   # skip append on same rerun
}.items():
    if k not in ss: ss[k] = v

# ═══════════════════════════════════════
# 3 . Load model + metadata
# ═══════════════════════════════════════
model       = load_model()
schema_meta = load_schema()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

# ═══════════════════════════════════════
# 4 . Helper functions
# ═══════════════════════════════════════
def label_risk(p):
    return "🟢 Low" if p < .30 else "🟡 Moderate" if p < .60 else "🔴 High"

def safe_stats(col):
    meta = schema_meta.get(col, {})
    lo, hi = float(meta.get("min", 0)), float(meta.get("max", 1))
    if lo == hi: hi += 1
    mean = float(meta.get("mean", (lo + hi) / 2))
    return lo, hi, mean

# ═══════════════════════════════════════
# 5 . UI header
# ═══════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown(
    "A decision-support tool for HR pros to predict attrition and "
    "understand the drivers via SHAP. Get clear probability, risk tier, "
    "and feature insights for single employees or bulk CSV uploads."
)

with st.expander("**How to use this app**"):
    st.markdown(
        "1. Fill in sidebar (or *Use Sample Data*).  \n"
        "2. Optionally upload a CSV for batch scoring.  \n"
        "3. Click **Run Prediction**.  \n"
        "4. Explore results & SHAP plots.  \n"
        "5. Download or clear prediction history."
    )

# ═══════════════════════════════════════
# 6 . Sidebar – inputs
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        key, tip = f"inp_{col}", tooltips.get(col.split("_")[0], "")
        if meta["dtype"] == "object":                     # dropdown
            opts = meta["options"]
            cur  = ss.get(key, opts[0] if opts else "")
            row[col] = st.sidebar.selectbox(col, opts, index=opts.index(cur),
                                            key=key, help=tip)
        else:                                             # numeric
            lo, hi, _ = safe_stats(col)
            cur  = float(ss.get(key, lo))
            cur  = min(max(cur, lo), hi)
            step = 1.0 if meta.get("discrete", False) else 0.1
            row[col] = st.sidebar.slider(col, lo, hi, value=cur, step=float(step),
                                         key=key, help=tip)
    return pd.DataFrame([row])

# --- Sample & Reset buttons --------------------------------------
sample_employee = {  # keep keys aligned with schema
    "Age": 32, "Attrition": "No", "Business Travel": "Travel_Rarely",
    "Daily Rate": 1100, "Department": "Research & Development",
    "Distance From Home": 8, "Education": "Bachelor's",
    "Education Field": "Life Sciences", "Environment Satisfaction": 3,
    "Gender": "Male", "Hourly Rate": 65, "Job Involvement": 3,
    "Job Level": 2, "Job Role": "Research Scientist", "Job Satisfaction": 2,
    "Marital Status": "Single", "Monthly Income": "5 000 – 5 999",
    "Monthly Rate": "10 000 – 14 999", "No. of Companies Worked": 2,
    "Over Time": "Yes", "Percent Salary Hike": 13, "Performance Rating": 3,
    "Relationship Satisfaction": 2, "Stock Option Level": 1,
    "Total Working Years": 10, "Training Times Last Year": 3,
    "Work Life Balance": 2, "Years At Company": 5,
    "Years In Current Role": 3, "Years Since Last Promotion": 1,
    "Years With Current Manager": 2,
}
def load_sample():
    for c, v in sample_employee.items():
        ss[f"inp_{c}"] = v
def reset_form():
    for c, meta in schema_meta.items():
        ss[f"inp_{c}"] = meta["options"][0] if meta["dtype"] == "object" else safe_stats(c)[0]

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form",    on_click=reset_form)

# ═══════════════════════════════════════
# 7 .  Data intake
# ═══════════════════════════════════════
uploaded = st.file_uploader("📂 Upload CSV (optional)", type="csv")
batch_mode = uploaded is not None
raw_df = pd.read_csv(uploaded) if batch_mode else sidebar_inputs()

# ═══════════════════════════════════════
# 8 .  Run / re-run control
# ═══════════════════════════════════════
run_now = st.sidebar.button("▶️ Run Prediction")
if run_now: ss.predicted = True          # remember till cleared / reset
if not ss.predicted:
    st.stop()

# ═══════════════════════════════════════
# 9 .  Encode data once
# ═══════════════════════════════════════
template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
            for c, m in schema_meta.items()}
X_full  = pd.concat([raw_df, pd.DataFrame([template])], ignore_index=True)
X_enc   = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc   = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

preds  = model.predict(X_enc)
probs  = model.predict_proba(X_enc)[:, 1]

# ═══════════════════════════════════════
# 10 .  Batch table + row picker
# ═══════════════════════════════════════
if batch_mode:
    tbl = raw_df.copy()
    tbl.insert(0, "Row", np.arange(1, len(tbl) + 1))
    tbl["Prediction"]    = np.where(preds == 1, "Yes", "No")
    tbl["Probability"]   = (probs * 100).round(1).astype(str) + " %"
    tbl["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    st.dataframe(tbl, use_container_width=True)

    row_label = st.selectbox(
        "Select employee row for explanation:",
        options=[f"{i}" for i in range(1, len(tbl) + 1)],
        index=0,
        key="row_select",
    )
    row_idx  = int(row_label) - 1
    X_user   = X_enc.iloc[[row_idx]]
    user_df  = raw_df.iloc[[row_idx]]
    pred, prob = preds[row_idx], probs[row_idx]
    risk = label_risk(prob)
else:
    X_user  = X_enc.iloc[[0]]
    user_df = raw_df.iloc[[0]]
    pred, prob = preds[0], probs[0]
    risk = label_risk(prob)

# ═══════════════════════════════════════
# 11 .  Results + SHAP
# ═══════════════════════════════════════
st.markdown("### 🎯 Prediction Results")
st.markdown(
    f"""
<div style='border:2px solid #eee;border-radius:10px;padding:20px;background:#f9f9f9;'>
  <div style='display:flex;justify-content:space-between;font-size:18px;'>
    <div><strong>Prediction</strong><br><span style='font-size:24px'>{'Yes' if pred else 'No'}</span></div>
    <div><strong>Probability</strong><br><span style='font-size:24px'>{prob:.1%}</span></div>
    <div><strong>Risk Category</strong><br><span style='font-size:24px'>{risk}</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("🔍 SHAP Explanations")
sv = explainer.shap_values(X_user)
if isinstance(sv, (list, tuple)): sv = sv[1]

fig, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig); plt.clf()

# … (decision, force, feature-impact plots omitted for brevity) …

# ═══════════════════════════════════════
# 12 .  Append to History (once per run)
# ═══════════════════════════════════════
if not ss.just_cleared:
    append_df         = user_df.copy()
    append_df["Prediction"]  = "Yes" if pred else "No"
    append_df["Probability"] = f"{prob:.1%}"
    append_df["Risk Category"] = risk
    append_df["Timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M")
    ss.history = pd.concat([ss.history, append_df], ignore_index=True)
ss.just_cleared = False

# ═══════════════════════════════════════
# 13 .  History display / download / clear
# ═══════════════════════════════════════
st.subheader("📜 Prediction History")
st.dataframe(ss.history, use_container_width=True)

csv_hist = ss.history.to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   file_name="prediction_history.csv",
                   mime="text/csv")

# — Clear button *after* download button —
if st.button("🗑️ Clear History", key="clear_history_bottom"):
    ss.history       = pd.DataFrame()
    ss.just_cleared  = True
    ss.predicted     = False
    st.rerun()
###################################################################
