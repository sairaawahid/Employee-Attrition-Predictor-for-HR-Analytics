import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# -- patch for legacy models saved with joblib.Bunch ---------------
# try:
#     from sklearn.utils import Bunch                    # type: ignore
#     sys.modules["joblib.Bunch"] = Bunch
# except Exception:
#     pass
# -----------------------------------------------------------------


st.set_page_config(
    page_title="Attrition Predictor",
    initial_sidebar_state="expanded"     # ← keeps sidebar open
)

# ═══════════════════════════════════════
# 1.  Cached resources
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
# 2.  Session-state keys
# ═══════════════════════════════════════
ss = st.session_state
defaults = {
    "history"      : pd.DataFrame(),
    "predicted"    : False,      # True after Run Prediction
    "just_cleared" : False,      # skip append right after clear
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# ═══════════════════════════════════════
# 3.  Load model + metadata
# ═══════════════════════════════════════
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

# ═══════════════════════════════════════
# 4.  Helpers
# ═══════════════════════════════════════
def label_risk(p):
    return "🟢 Low" if p < .30 else "🟡 Moderate" if p < .60 else "🔴 High"

def safe_stats(col):
    meta = schema_meta.get(col, {})
    lo, hi = float(meta.get("min", 0)), float(meta.get("max", 1))
    if lo == hi: hi += 1
    return lo, hi, float(meta.get("mean", (lo + hi) / 2))

# ═══════════════════════════════════════
# 5.  Header
# ═══════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown(
    "Predict attrition for a single employee or a whole CSV upload and "
    "instantly explore **SHAP** explanations.  "
    "Results are saved in _Prediction History_ until you clear them."
)

# ═══════════════════════════════════════
# 6.  Sidebar  – attributes
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")
        if meta["dtype"] == "object":
            opts = meta["options"]
            row[col] = st.sidebar.selectbox(col, opts,
                                            index=opts.index(ss.get(key, opts[0])),
                                            key=key, help=tip)
        else:
            lo, hi, _ = safe_stats(col)
            cur = float(ss.get(key, lo))
            cur = min(max(cur, lo), hi)
            step = 1.0 if meta.get("discrete", False) else 0.1
            row[col] = st.sidebar.slider(col, lo, hi, value=cur,
                                         step=float(step), key=key, help=tip)
    return pd.DataFrame([row])

# -- Sample & Reset buttons ---------------------------------------
sample_employee = {
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
    ss.predicted = False          # no auto-save
def reset_form():
    for c, meta in schema_meta.items():
        ss[f"inp_{c}"] = meta["options"][0] if meta["dtype"] == "object" else safe_stats(c)[0]
    ss.predicted = False          # no auto-save

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form",      on_click=reset_form)

# ═══════════════════════════════════════
# 7.  Data intake
# ═══════════════════════════════════════
uploaded     = st.file_uploader("📂 Upload CSV (optional)", type="csv")
batch_mode   = uploaded is not None
raw_df       = pd.read_csv(uploaded) if batch_mode else sidebar_inputs()

# ═══════════════════════════════════════
# 8.  Prediction trigger
# ═══════════════════════════════════════
clicked_run   = st.sidebar.button("▶️ Run Prediction")
append_needed = clicked_run            # append only on button click
if clicked_run:        # remember that user has run once
    ss.predicted    = True
    ss.just_cleared = False            # allow new history rows

if not ss.predicted:
    st.stop()

# ═══════════════════════════════════════
# 9.  Encode & predict once
# ═══════════════════════════════════════
template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
            for c, m in schema_meta.items()}
X_full = pd.concat([raw_df, pd.DataFrame([template])], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

preds, probs = model.predict(X_enc), model.predict_proba(X_enc)[:, 1]

# ═══════════════════════════════════════
# 10.  Batch table + row picker
# ═══════════════════════════════════════
if batch_mode:
    tbl = raw_df.copy()
    tbl.insert(0, "Row", np.arange(1, len(tbl)+1))
    tbl["Prediction"]    = np.where(preds == 1, "Yes", "No")
    tbl["Probability"]   = (probs*100).round(1).astype(str)+" %"
    tbl["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    st.dataframe(tbl, use_container_width=True)

    row_num = st.selectbox("Select employee row for explanation",
                           options=list(range(1, len(tbl)+1)), index=0)
    idx     = row_num-1
else:
    idx = 0

X_user, user_df = X_enc.iloc[[idx]], raw_df.iloc[[idx]]
pred, prob      = preds[idx], probs[idx]
risk            = label_risk(prob)

# ═══════════════════════════════════════
# 11.  Results + SHAP
# ═══════════════════════════════════════
st.markdown("### 🔮 Prediction Result")
st.markdown(
    f"<div style='border:2px solid #eee;border-radius:10px;padding:20px;background:#f9f9f9'>"
    f"<b>Prediction:</b> {'Yes' if pred else 'No'} &nbsp;&nbsp;|&nbsp;&nbsp;"
    f"<b>Probability:</b> {prob:.1%} &nbsp;&nbsp;|&nbsp;&nbsp;"
    f"<b>Risk:</b> {risk}</div>", unsafe_allow_html=True
)

sv = explainer.shap_values(X_user)
if isinstance(sv, (list, tuple)):
    sv = sv[1]

with st.expander("SHAP summary (global impact)", expanded=True):
    fig, _ = plt.subplots()
    shap.summary_plot(sv, X_user, show=False)
    st.pyplot(fig); plt.clf()

# (other SHAP plots can be added similarly)

# ═══════════════════════════════════════
# 12.  Append to history ONLY when needed
# ═══════════════════════════════════════
if append_needed and not ss.just_cleared:
    hrow = user_df.copy()
    hrow["Prediction"]  = "Yes" if pred else "No"
    hrow["Probability"] = f"{prob:.1%}"
    hrow["Risk Category"] = risk
    hrow["Timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M")
    ss.history = pd.concat([ss.history, hrow], ignore_index=True)

# ═══════════════════════════════════════
# 13.  History panel
# ═══════════════════════════════════════
st.subheader("📜 Prediction History")
if ss.history.empty:
    st.info("No predictions yet.")
else:
    st.dataframe(ss.history, use_container_width=True)

csv_hist = ss.history.to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   file_name="prediction_history.csv", mime="text/csv")

if st.button("🗑️ Clear History"):
    ss.history     = pd.DataFrame()
    ss.just_cleared = True    # suppress immediate append
    ss.predicted   = False    # user must click Run again
    st.rerun()
###################################################################
