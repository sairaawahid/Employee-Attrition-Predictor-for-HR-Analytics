import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, json, matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ═════════════════════════════════════════════════════════════
# 1.  CACHED LOADERS
# ═════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    """employee_schema.json created from original IBM dataset"""
    return json.loads(Path("employee_schema.json").read_text())

@st.cache_data
def load_tooltips():
    fp = Path("feature_tooltips.json")
    return json.loads(fp.read_text()) if fp.exists() else {}

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)          # no hashing on model

# ═════════════════════════════════════════════════════════════
# 2.  INITIALISE
# ═════════════════════════════════════════════════════════════
model       = load_model()
schema_meta = load_schema()           # {col: {dtype, [options|min|max|mean]}}
tooltips    = load_tooltips()
explainer   = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ═════════════════════════════════════════════════════════════
# 3.  HELPERS
# ═════════════════════════════════════════════════════════════
def risk_label(p):
    return "🟢 Low" if p < .30 else "🟡 Moderate" if p < .60 else "🔴 High"

def numeric_stats(col):
    meta = schema_meta[col]
    cmin = meta.get("min",   0.0)
    cmax = meta.get("max",   1.0)
    cmean= meta.get("mean", (cmin + cmax) / 2)
    # make sure min < max for Streamlit slider
    if cmin == cmax:
        cmin, cmax = cmin - 1, cmax + 1
    return cmin, cmax, cmean

# ═════════════════════════════════════════════════════════════
# 4.  HEADER / GUIDE
# ═════════════════════════════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk & interpret results with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
1. Use the **sidebar** to enter one employee’s data – or click **Use Sample
   Data** (quick demo) / **Reset Form**.  
2. **Optional**: upload a CSV for **batch scoring** – a row selector appears for
   individual inspection.  
3. Review **risk cards**, **global & local SHAP charts**, and the **Interactive
   Feature Impact** viewer.  
4. **Prediction history** is saved; download or clear anytime.
"""
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ═════════════════════════════════════════════════════════════
# 5.  SIDEBAR INPUTS
# ═════════════════════════════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs() -> pd.DataFrame:
    row = {}
    for col, meta in schema_meta.items():
        tip = tooltips.get(col.split("_")[0], "")
        key = f"inp_{col}"

        if meta["dtype"] == "object":
            opts = meta["options"]
            cur  = st.session_state.get(key, opts[0])
            if cur not in opts:
                cur = opts[0]
            row[col] = st.sidebar.selectbox(col, opts, index=opts.index(cur),
                                            key=key, help=tip)
        else:
            cmin, cmax, cmean = numeric_stats(col)
            val  = float(st.session_state.get(key, cmean))
            val  = min(max(val, cmin), cmax)
            row[col] = st.sidebar.slider(col, cmin, cmax, val, key=key, help=tip)
    return pd.DataFrame([row])

# ═════════════════════════════════════════════════════════════
# 6.  SAMPLE & RESET
# ═════════════════════════════════════════════════════════════
sample_employee = {
    "Age": 32, "Gender": "Male", "Department": "Research & Development",
    "BusinessTravel": "Travel_Rarely", "MonthlyIncome": 5200,
}
def use_sample():
    for col, val in sample_employee.items():
        st.session_state[f"inp_{col}"] = val
def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        st.session_state[key] = meta["options"][0] if meta["dtype"] == "object" \
                                else numeric_stats(col)[2]

st.sidebar.button("🧭 Use Sample Data", on_click=use_sample)
st.sidebar.button("🔄 Reset Form",      on_click=reset_form)

# ═════════════════════════════════════════════════════════════
# 7.  DATAFRAME SOURCE (sidebar or CSV)
# ═════════════════════════════════════════════════════════════
if uploaded_file is not None:
    batch_df   = pd.read_csv(uploaded_file)
    st.session_state["batch_df"] = batch_df.copy()
    batch_mode = True
else:
    batch_mode = False

if batch_mode:
    # Row selector for individual inspection
    idx = st.number_input("🔢 Row to inspect", 0, len(st.session_state["batch_df"]) - 1, 0)
    raw_df = st.session_state["batch_df"].iloc[[idx]].reset_index(drop=True)
else:
    raw_df = sidebar_inputs()

# ═════════════════════════════════════════════════════════════
# 8.  ONE–HOT TO MATCH MODEL
# ═════════════════════════════════════════════════════════════
template = {c: (schema_meta[c]["options"][0] if schema_meta[c]["dtype"] == "object" else 0)
            for c in schema_meta}
schema_df = pd.DataFrame([template])

X_full = pd.concat([raw_df, schema_df], ignore_index=True)
X_enc  = pd.get_dummies(X_full)[: len(raw_df)]
X_user = X_enc.iloc[[0]]

# ═════════════════════════════════════════════════════════════
# 9.  PREDICT & METRICS
# ═════════════════════════════════════════════════════════════
pred   = model.predict(X_user)[0]
prob   = model.predict_proba(X_user)[0, 1]
risk   = risk_label(prob)

st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk)

# ═════════════════════════════════════════════════════════════
# 10.  SHAP – GLOBAL & LOCAL
# ═════════════════════════════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
sv = explainer.shap_values(X_user)
if isinstance(sv, list): sv = sv[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig_b, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig_b); plt.clf()

st.markdown("### 🧭 Decision Path")
fig_d, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(fig_d); plt.clf()

st.markdown("### 🎯 Local Force Plot")
try:
    fig_f = shap.plots.force(explainer.expected_value, sv[0], X_user.iloc[0],
                             matplotlib=True, show=False)
    st.pyplot(fig_f)
except Exception:
    st.info("Force plot unavailable – showing waterfall instead.")
    fig_f, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=sv[0], base_values=explainer.expected_value,
                         data=X_user.iloc[0]),
        max_display=15, show=False)
    st.pyplot(fig_f)

# ═════════════════════════════════════════════════════════════
# 11.  INTERACTIVE FEATURE IMPACT
# ═════════════════════════════════════════════════════════════
st.markdown("### 🔎 Interactive Feature Impact")
chosen = st.selectbox("Select feature", X_user.columns)
col_idx = X_user.columns.get_loc(chosen)

fig_i, _ = plt.subplots()
shap.bar_plot(np.array([sv[0][col_idx]]),
              feature_names=[chosen],
              max_display=1,
              show=False)
st.pyplot(fig_i); plt.clf()

# ═════════════════════════════════════════════════════════════
# 12.  BATCH SUMMARY (if CSV)
# ═════════════════════════════════════════════════════════════
if batch_mode:
    preds = model.predict(pd.get_dummies(X_full)[: len(st.session_state["batch_df"])])
    probs = model.predict_proba(pd.get_dummies(X_full)[: len(st.session_state["batch_df"])])[:, 1]
    out   = st.session_state["batch_df"].copy()
    out["Prediction"]    = np.where(preds == 1, "Yes", "No")
    out["Probability"]   = (probs * 100).round(1).astype(str) + " %"
    out["Risk Category"] = [risk_label(p) for p in probs]
    st.markdown("### 📑 Batch Prediction Summary")
    st.dataframe(out)

# ═════════════════════════════════════════════════════════════
# 13.  HISTORY
# ═════════════════════════════════════════════════════════════
now = datetime.now().strftime("%Y-%m-%d %H:%M")
hist = raw_df.copy()
hist["Prediction"]    = "Yes" if pred else "No"
hist["Probability"]   = f"{prob:.1%}"
hist["Risk Category"] = risk
hist["Timestamp"]     = now
st.session_state["history"] = pd.concat([st.session_state["history"], hist],
                                        ignore_index=True)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])

csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   file_name="prediction_history.csv",
                   mime="text/csv")

if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
