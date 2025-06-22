import streamlit as st

# ──────────────────────────────────────────────────────────────
# Back-compat shim – always provide st.experimental_rerun()
# (new Streamlit versions renamed it to st.rerun())
# ──────────────────────────────────────────────────────────────
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    st.experimental_rerun = st.rerun          # type: ignore

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime


# ───────── Streamlit config – keep sidebar open ─────────────
st.set_page_config(page_title="Attrition Predictor",
                   initial_sidebar_state="expanded")

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
defaults = {
    "history"        : pd.DataFrame(),
    "predicted"      : False,   # at least one prediction run in this session
    "append_pending" : False,   # append *once* immediately after Run Prediction
    "load_sample"    : False,
}
for k, v in defaults.items():
    ss.setdefault(k, v)

# ═══════════════════════════════════════
# 3 . Load model & metadata
# ═══════════════════════════════════════
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

# ═══════════════════════════════════════
# 4 . Helper functions
# ═══════════════════════════════════════
def label_risk(p: float) -> str:
    if p < 0.30: return "🟢 Low"
    if p < 0.60: return "🟡 Moderate"
    return "🔴 High"

def safe_stats(col: str):
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

# ═══════════════════════════════════════
# 6 . Sidebar – inputs
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
    row = {}
    for col, meta in schema_meta.items():
        key, tip = f"inp_{col}", tooltips.get(col.split("_")[0], "")
        if meta["dtype"] == "object":                     # dropdown
            opts = meta["options"]
            cur  = ss.get(key, opts[0] if opts else "")
            row[col] = st.sidebar.selectbox(col, opts,
                                            index=opts.index(cur),
                                            key=key, help=tip)
        else:                                             # numeric
            lo, hi, _ = safe_stats(col)
            cur  = float(ss.get(key, lo))
            cur  = min(max(cur, lo), hi)
            step = 1.0 if meta.get("discrete", False) else 0.1
            row[col] = st.sidebar.slider(col, lo, hi,
                                         value=cur,
                                         step=float(step),
                                         key=key, help=tip)
    return pd.DataFrame([row])

# --- Sample & Reset buttons ------------------------------------

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

# ----- Helper: make sample dict complete -----
def _complete_sample_dict():
    """Return a dict that has *every* column in the schema.
       Any missing key gets the schema default."""
    full = {}
    for col, meta in schema_meta.items():
        if col in sample_employee:                 # value you provided
            full[col] = sample_employee[col]
        elif meta["dtype"] == "object":            # default dropdown
            full[col] = meta["options"][0]
        else:                                      # default numeric = min
            full[col] = safe_stats(col)[0]
    return full
    
def load_sample():
    for col, val in _complete_sample_dict().items():
        ss[f"inp_{col}"] = val
    ss["load_sample"] = True

def reset_form():
    for c, m in schema_meta.items():
        ss[f"inp_{c}"] = m["options"][0] if m["dtype"] == "object" else safe_stats(c)[0]

st.sidebar.button("Use Sample Data", on_click=load_sample)
if ss.load_sample:
    ss.load_sample = False       # reset it
    st.experimental_rerun()      # safe rerun outside callback
st.sidebar.button("🔄 Reset Form",    on_click=reset_form)

# ═══════════════════════════════════════
# 7 .  Data intake
# ═══════════════════════════════════════
uploaded     = st.file_uploader("📂 Upload CSV (optional)", type="csv")
batch_mode   = uploaded is not None
raw_df       = pd.read_csv(uploaded) if batch_mode else sidebar_inputs()

# ═══════════════════════════════════════
# 8 .  Run Prediction control
# ═══════════════════════════════════════
if st.sidebar.button("Run Prediction"):
    ss.predicted      = True
    ss.append_pending = True

# ═══════════════════════════════════════
# Attribution Footer
# ═══════════════════════════════════════
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style='font-size: 12px; color: #6c757d; text-align: center; padding-top: 10px;'>
        © 2025 <strong>Sairaawahid<strong>. All rights reserved.<br>
        If you use or adapt this project, please give credit by linking to the 
        <a href="https://github.com/sairaawahid/Employee-Attrition-Predictor-for-HR-Analytics" target="_blank">
        GitHub repository</a>.
    </div>
    """,
    unsafe_allow_html=True
)

if not ss.predicted:             # haven’t run yet – do nothing else
    st.stop()

# ═══════════════════════════════════════
# 9 .  Encode data & predict
# ═══════════════════════════════════════
template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
            for c, m in schema_meta.items()}
X_full = pd.concat([raw_df, pd.DataFrame([template])], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

preds  = model.predict(X_enc)
probs  = model.predict_proba(X_enc)[:, 1]

# ═══════════════════════════════════════
# 10 .  Batch table + picker
# ═══════════════════════════════════════
if batch_mode:
    tbl = raw_df.copy()
    tbl.insert(0, "Row", np.arange(1, len(tbl)+1))
    tbl["Prediction"]    = np.where(preds==1, "Yes", "No")
    tbl["Probability"]   = (probs*100).round(1).astype(str)+" %"
    tbl["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    st.dataframe(tbl, use_container_width=True)

    sel_row_lbl = st.selectbox(
        "**Select employee row for explanation**",
        [str(i) for i in range(1, len(tbl)+1)],
        index=0, key="row_select"
    )
    row_idx = int(sel_row_lbl) - 1
else:
    row_idx = 0

# Data for single explanation
X_user  = X_enc.iloc[[row_idx]]
user_df = raw_df.iloc[[row_idx]]
pred, prob = preds[row_idx], probs[row_idx]
risk = label_risk(prob)

# ═══════════════════════════════════════
# 11 .  Results + SHAP (unchanged UI)
# ═══════════════════════════════════════
st.markdown("### Prediction Results")
st.markdown(
    f"""
<div style='border:2px solid #eee;border-radius:10px;padding:20px;background:#f9f9f9;'>
  <div style='display:flex;justify-content:space-between;font-size:18px;'>
    <div><strong>Prediction</strong><br><span style='font-size:24px'>{'Yes' if pred else 'No'}</span></div>
    <div><strong>Probability</strong><br><span style='font-size:24px'>{prob:.1%}</span></div>
    <div><strong>Risk&nbsp;Category</strong><br><span style='font-size:24px'>{risk}</span></div>
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

st.markdown("### 1. Global Impact — Beeswarm")
st.info("This plot shows which features **had the highest overall impact** "
        "on the model’s prediction for this employee. Longer bars = stronger effect. "
        "Colors indicate whether the value pushed the prediction higher (red) or lower (blue).")
fig_bsw, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig_bsw); plt.clf()

st.markdown("### 2. Decision Path")
st.info("This plot explains the **sequence of contributions** each feature made, "
        "starting from the model’s baseline prediction. Features that increased or "
        "decreased the risk are shown from left to right, helping you follow the model’s logic.")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(fig_dec); plt.clf()

st.markdown("### 3. Local Force Plot")
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

st.markdown("### 4. Interactive Feature Impact")
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
# 12 .  Append to history exactly once
# ═══════════════════════════════════════
if ss.append_pending:
    rec = user_df.copy()
    rec["Prediction"]    = "Yes" if pred else "No"
    rec["Probability"]   = f"{prob:.1%}"
    rec["Risk Category"] = risk
    rec["Timestamp"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
    ss.history = pd.concat([ss.history, rec], ignore_index=True)

ss.append_pending = False
ss.just_cleared   = False

# ═══════════════════════════════════════
# 13 .  History display / download / clear
# ═══════════════════════════════════════
st.subheader("📜 Prediction History")
st.dataframe(ss.history, use_container_width=True)

csv_hist = ss.history.to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   file_name="prediction_history.csv",
                   mime="text/csv")

# -- Clear History button directly under download -----------------
if st.button("🗑️ Clear History", key="clear_history"):
    ss.history       = pd.DataFrame()
    ss.predicted     = False
    ss.append_pending = False
    st.experimental_rerun()


