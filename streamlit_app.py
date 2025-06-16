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

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model       = load_model()
schema_meta = load_schema()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────
def label_risk(p):
    if p < .30: return "🟢 Low"
    if p < .60: return "🟡 Moderate"
    return "🔴 High"

def safe_stats(col):
    m = schema_meta[col]
    return m.get("min", 0.0), m.get("max", 1.0), m.get("mean", 0.5)

# ──────────────────────────────────────────────────────────────
# 4.  HEADER & FILE UPLOAD
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown("""
    1. Use the **sidebar** to input features or click **Use Sample Data** / **Reset Form**.  
    2. Or upload a **CSV** file for batch predictions.  
    3. See prediction, risk, SHAP plots, and explanations.  
    4. Select a row to view feature impacts interactively.
    """)

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ──────────────────────────────────────────────────────────────
# 5.  SIDEBAR INPUT FORM
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        tip = tooltips.get(col.split("_")[0], "")
        key = f"inp_{col}"

        if meta["dtype"] == "object":
            opts = meta.get("options", ["Unknown"])
            val = st.session_state.get(key, opts[0])
            val = val if val in opts else opts[0]
            row[col] = st.sidebar.selectbox(col, opts, index=opts.index(val), key=key, help=tip)
        else:
            cmin, cmax, cmean = safe_stats(col)
            val = float(st.session_state.get(key, cmean))
            val = max(min(val, cmax), cmin)
            row[col] = st.sidebar.slider(col, cmin, cmax, val, key=key, help=tip)
    return pd.DataFrame([row])

# ──────────────────────────────────────────────────────────────
# 6.  SAMPLE + RESET BUTTONS
# ──────────────────────────────────────────────────────────────
sample_employee = {
    "Age": 32,
    "Gender": "Male",
    "Department": "Research & Development",
    "BusinessTravel": "Travel_Rarely",
    "MonthlyIncome": 5200
}

def load_sample():
    for col, val in sample_employee.items():
        st.session_state[f"inp_{col}"] = val

def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        if meta["dtype"] == "object":
            st.session_state[key] = meta["options"][0]
        else:
            st.session_state[key] = meta["mean"]

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form",     on_click=reset_form)

# ──────────────────────────────────────────────────────────────
# 7.  COLLECT INPUT
# ──────────────────────────────────────────────────────────────
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# ──────────────────────────────────────────────────────────────
# 8.  ONE-HOT ENCODE + ALIGN
# ──────────────────────────────────────────────────────────────
template = {col: (meta["options"][0] if meta["dtype"] == "object" else 0)
            for col, meta in schema_meta.items()}
base_df = pd.DataFrame([template])

X_full = pd.concat([raw_df, base_df], ignore_index=True)
X_enc = pd.get_dummies(X_full).iloc[:len(raw_df)]

# ──────────────────────────────────────────────────────────────
# 9.  ROW SELECTION FOR BATCH
# ──────────────────────────────────────────────────────────────
selected_row = 0
if batch_mode:
    selected_row = st.selectbox("🔍 Select Row to Inspect", range(len(raw_df)), format_func=lambda x: f"Row {x+1}")
X_user = X_enc.iloc[[selected_row]]

# ──────────────────────────────────────────────────────────────
# 10.  PREDICT + METRICS
# ──────────────────────────────────────────────────────────────
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk = label_risk(prob)

st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk)

# ──────────────────────────────────────────────────────────────
# 11.  SHAP PLOTS
# ──────────────────────────────────────────────────────────────
shap_vals = explainer.shap_values(X_user)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

st.subheader("🔍 SHAP Explanations")
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
    fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0],
                            matplotlib=True, show=False)
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

# ──────────────────────────────────────────────────────────────
# 12.  INTERACTIVE IMPACT VIEWER
# ──────────────────────────────────────────────────────────────
st.subheader("📊 Feature Impact Viewer")
feat = st.selectbox("Select a feature", X_user.columns)
bar_df = pd.DataFrame({
    "Feature": [feat],
    "SHAP Value": [shap_vals[0][X_user.columns.get_loc(feat)]]
})
st.bar_chart(bar_df.set_index("Feature"))

# ──────────────────────────────────────────────────────────────
# 13.  BATCH SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
if batch_mode:
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    out = raw_df.copy()
    out["Prediction"] = np.where(preds == 1, "Yes", "No")
    out["Probability"] = (probs * 100).round(1).astype(str) + " %"
    out["Risk Category"] = [label_risk(p) for p in probs]
    st.subheader("📑 Batch Prediction Summary")
    st.dataframe(out)

# ──────────────────────────────────────────────────────────────
# 14.  PREDICTION HISTORY
# ──────────────────────────────────────────────────────────────
row = raw_df.iloc[[selected_row]].copy()
row["Prediction"] = "Yes" if pred else "No"
row["Probability"] = f"{prob:.1%}"
row["Risk Category"] = risk
row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat(
    [st.session_state["history"], row], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist, "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
