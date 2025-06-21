import sys, json, joblib, shap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────
#  Quick patch: old sklearn models saved with joblib.Bunch
# ──────────────────────────────────────────────────────────────
try:
    from sklearn.utils import Bunch          # type: ignore
    sys.modules["joblib.Bunch"] = Bunch
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
#  Page config – keep sidebar open
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attrition Predictor",
                   initial_sidebar_state="expanded")

# --------- Streamlit version-agnostic rerun helper ----------
def _safe_rerun():
    """Call st.rerun() on new Streamlit, or experimental_rerun() on <1.25."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
        
# ═════════════════════════════════════════════════════════════
# 1 ▸  Cached resources
# ═════════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════════
# 2 ▸  Session-state keys
# ═════════════════════════════════════════════════════════════
ss = st.session_state
defaults = {
    "history"       : pd.DataFrame(),  # prediction log
    "predicted"     : False,           # True once Run-Prediction clicked
    "append_pending": False,           # log row on *this* rerun
    "just_cleared"  : False,           # stop first append after clear
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# ═════════════════════════════════════════════════════════════
# 3 ▸  Load model + metadata
# ═════════════════════════════════════════════════════════════
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

# ═════════════════════════════════════════════════════════════
# 4 ▸  Helpers
# ═════════════════════════════════════════════════════════════
def label_risk(p):
    return "🟢 Low" if p < .30 else "🟡 Moderate" if p < .60 else "🔴 High"

def safe_stats(col):
    meta = schema_meta.get(col, {})
    lo, hi = float(meta.get("min", 0)), float(meta.get("max", 1))
    if lo == hi:
        hi += 1
    mean = float(meta.get("mean", (lo + hi) / 2))
    return lo, hi, mean

# ═════════════════════════════════════════════════════════════
# 5 ▸  Header
# ═════════════════════════════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown(
    "Predict attrition risk and explore **SHAP** explanations – for a "
    "single employee or an uploaded CSV – while keeping a running "
    "history of your analyses."
)

with st.expander("**How to use this app**", expanded=False):
    st.markdown(
        "1. Fill in the sidebar (or click **Use Sample Data**).  \n"
        "2. Optionally upload a CSV for batch scoring.  \n"
        "3. Click **Run Prediction**. Results & SHAP appear.  \n"
        "4. All predictions are stored in *Prediction History* "
        "until you click **Clear History**."
    )

# ═════════════════════════════════════════════════════════════
# 6 ▸  Sidebar – employee attributes
# ═════════════════════════════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
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

# --- sample + reset ------------------------------------------------
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
    ss.predicted = False
    ss.append_pending = False
def reset_form():
    for c, meta in schema_meta.items():
        ss[f"inp_{c}"] = meta["options"][0] if meta["dtype"] == "object" else safe_stats(c)[0]
    ss.predicted = False
    ss.append_pending = False

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form",      on_click=reset_form)

# ═════════════════════════════════════════════════════════════
# 7 ▸  Data intake
# ═════════════════════════════════════════════════════════════
uploaded  = st.file_uploader("📂 Upload CSV (optional)", type="csv")
batch     = uploaded is not None
raw_df    = pd.read_csv(uploaded) if batch else sidebar_inputs()

# ═════════════════════════════════════════════════════════════
# 8 ▸  Run Prediction button
# ═════════════════════════════════════════════════════════════
run_clicked = st.sidebar.button("▶️ Run Prediction")

if run_clicked:
    ss.predicted      = True            # show results
    ss.append_pending = True            # log once
    ss.just_cleared   = False           # allow logging

if not ss.predicted:
    st.stop()

# ═════════════════════════════════════════════════════════════
# 9 ▸  Encode & model predict
# ═════════════════════════════════════════════════════════════
template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
            for c, m in schema_meta.items()}
X_full = pd.concat([raw_df, pd.DataFrame([template])], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

preds  = model.predict(X_enc)
probs  = model.predict_proba(X_enc)[:, 1]

# ═════════════════════════════════════════════════════════════
# 10 ▸  Batch table & row picker
# ═════════════════════════════════════════════════════════════
if batch:
    tbl = raw_df.copy()
    tbl.insert(0, "Row", np.arange(1, len(tbl)+1))
    tbl["Prediction"]    = np.where(preds == 1, "Yes", "No")
    tbl["Probability"]   = (probs*100).round(1).astype(str)+" %"
    tbl["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    st.dataframe(tbl, use_container_width=True)

    picked = st.selectbox("Select row for detailed explanation",
                          options=list(range(1, len(tbl)+1)),
                          index=0, key="row_picker")
    idx = picked - 1
else:
    idx = 0

X_user, user_df = X_enc.iloc[[idx]], raw_df.iloc[[idx]]
pred, prob      = preds[idx], probs[idx]
risk            = label_risk(prob)

# ═════════════════════════════════════════════════════════════
# 11 ▸  Results + SHAP (unchanged styling)
# ═════════════════════════════════════════════════════════════
st.markdown("### 🎯 Prediction Results")
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
st.info("These plots show which features push the prediction higher or lower.")

sv = explainer.shap_values(X_user)
if isinstance(sv, (list, tuple)):
    sv = sv[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig_bsw, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig_bsw); plt.clf()

st.markdown("### 🧭 Decision Path")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(fig_dec); plt.clf()

st.markdown("### 🎯 Local Force Plot")
try:
    fig_force = shap.plots.force(explainer.expected_value, sv[0],
                                 X_user.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig_force)
except Exception:
    fig_alt, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=sv[0], base_values=explainer.expected_value,
                         data=X_user.iloc[0]),
        max_display=15, show=False)
    st.pyplot(fig_alt)

st.markdown("### 🔎 Interactive Feature Impact")
feature = st.selectbox("Choose feature", X_user.columns, key="feat_sel")
idx_feat = X_user.columns.get_loc(feature)
fig_bar, _ = plt.subplots()
shap.bar_plot(np.array([sv[0][idx_feat]]), feature_names=[feature],
              max_display=1, show=False)
st.pyplot(fig_bar); plt.clf()

# ═════════════════════════════════════════════════════════════
# 12 ▸  Append to history once per click
# ═════════════════════════════════════════════════════════════
if ss.append_pending and not ss.just_cleared:
    log_row = user_df.copy()
    log_row["Prediction"]    = "Yes" if pred else "No"
    log_row["Probability"]   = f"{prob:.1%}"
    log_row["Risk Category"] = risk
    log_row["Timestamp"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
    ss.history = pd.concat([ss.history, log_row], ignore_index=True)

ss.append_pending = False   # consumed (or skipped)

# ═════════════════════════════════════════════════════════════
# 13 ▸  History section (download + clear)
# ═════════════════════════════════════════════════════════════
st.subheader("📜 Prediction History")
if ss.history.empty:
    st.info("No predictions yet.")
else:
    st.dataframe(ss.history, use_container_width=True)

csv_hist = ss.history.to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   file_name="prediction_history.csv",
                   mime="text/csv")

# -- Clear History button directly under download -----------------
if st.button("🗑️ Clear History", key="clear_history"):
    ss.history       = pd.DataFrame()
    ss.just_cleared  = True
    ss.predicted     = False
    ss.append_pending = False
    _safe_rerun()

