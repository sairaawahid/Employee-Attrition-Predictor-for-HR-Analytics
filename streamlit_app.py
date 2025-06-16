import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt, json
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
# 2.  Initialisation
# ═══════════════════════════════════════
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ═══════════════════════════════════════
# 3.  Helper utilities
# ═══════════════════════════════════════
def label_risk(p):
    if p < .30: return "🟢 Low"
    if p < .60: return "🟡 Moderate"
    return "🔴 High"

def safe_stats(col):
    meta = schema_meta.get(col, {})
    cmin = meta.get("min", 0.0)
    cmax = meta.get("max", 1.0)
    cmean = meta.get("mean", (cmin + cmax) / 2)
    if cmin == cmax:
        cmax = cmin + 1
    return float(cmin), float(cmax), float(cmean)

# ═══════════════════════════════════════
# 4.  UI Header
# ═══════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown("""
    • **Sidebar** to enter attributes (or **Use Sample Data** / **Reset Form**).  
    • Upload a CSV for **batch scoring**.  
    • Select any row for detailed SHAP inspection.  
    • View global & local SHAP plots and download prediction history.
    """)

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ═══════════════════════════════════════
# 5.  Sidebar widgets
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")

        if meta["dtype"] == "object":
            opts   = meta.get("options", ["Unknown"])
            curval = st.session_state.get(key, opts[0])
            if curval not in opts:
                curval = opts[0]
            row[col] = st.sidebar.selectbox(col, opts,
                                            index=opts.index(curval),
                                            key=key, help=tip)
        else:
            cmin, cmax, cmean = safe_stats(col)
            curval = float(st.session_state.get(key, cmean))
            curval = min(max(curval, cmin), cmax)   # clamp
            if abs(cmax - cmin) < 1e-9:
                row[col] = st.sidebar.number_input(col, value=curval,
                                                   key=key, help=tip)
            else:
                row[col] = st.sidebar.slider(col, cmin, cmax, curval,
                                             key=key, help=tip)
    return pd.DataFrame([row])

# ═══════════════════════════════════════
# 6.  Sample & Reset buttons
# ═══════════════════════════════════════
sample_employee = {
    "Age": 35,
    "Gender": "Male",
    "Department": "Research & Development",
    "BusinessTravel": "Travel_Rarely",
    "MonthlyIncome": 5500,
}

def load_sample():
    for col, val in sample_employee.items():
        if isinstance(val, (int, float)) and schema_meta[col]["dtype"] != "object":
            cmin, cmax, _ = safe_stats(col)
            val = max(min(val, cmax), cmin)
        st.session_state[f"inp_{col}"] = val

def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        if meta["dtype"] == "object":
            st.session_state[key] = meta.get("options", ["Unknown"])[0]
        else:
            st.session_state[key] = safe_stats(col)[2]

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form", on_click=reset_form)

# ═══════════════════════════════════════
# 7.  Gather user or batch data
# ═══════════════════════════════════════
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# ═══════════════════════════════════════
# 8.  Prepare data for model
# ═══════════════════════════════════════
template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
            for c, m in schema_meta.items()}
schema_df = pd.DataFrame([template])

X_full = pd.concat([raw_df, schema_df], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

# ═══════════════════════════════════════
# 9.  Batch prediction (if any)
# ═══════════════════════════════════════
if batch_mode:
    preds  = model.predict(X_enc)
    probs  = model.predict_proba(X_enc)[:, 1]
    raw_df["Prediction"]    = np.where(preds == 1, "Yes", "No")
    raw_df["Probability"]   = (probs*100).round(1).astype(str) + " %"
    raw_df["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    sel_row = st.number_input("Row to inspect (1-based)",
                              min_value=1, max_value=len(raw_df), value=1)
    st.dataframe(raw_df)
    X_user  = X_enc.iloc[[sel_row-1]]
    user_df = raw_df.iloc[[sel_row-1]]
else:
    X_user  = X_enc.iloc[[0]]
    user_df = raw_df.iloc[[0]]

# ═══════════════════════════════════════
# 10.  Single prediction display
# ═══════════════════════════════════════
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk = label_risk(prob)

st.subheader("🎯 Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction",   "Yes" if pred else "No")
c2.metric("Probability",  f"{prob:.1%}")
c3.metric("Risk Category", risk)

# ═══════════════════════════════════════
# 11.  SHAP explanations
# ═══════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
sv = explainer.shap_values(X_user)
if isinstance(sv, list):
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
    # st.warning("⚠️ Force plot not supported in this environment.")

st.markdown("### 🔎 Interactive Feature Impact")
feature = st.selectbox("Choose feature", X_user.columns, key="feat_sel")
idx     = X_user.columns.get_loc(feature)
single_val_arr = np.array([sv[0][idx]])

fig_bar, _ = plt.subplots()
shap.bar_plot(
    shap_values=single_val_arr,
    feature_names=[feature],
    max_display=1,
    show=False,
)
st.pyplot(fig_bar); plt.clf()

# ═══════════════════════════════════════
# 12.  History (append & display)
# ═══════════════════════════════════════
user_df = user_df.copy()
user_df["Prediction"]    = "Yes" if pred else "No"
user_df["Probability"]   = f"{prob:.1%}"
user_df["Risk Category"] = risk
user_df["Timestamp"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat(
    [st.session_state["history"], user_df], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
