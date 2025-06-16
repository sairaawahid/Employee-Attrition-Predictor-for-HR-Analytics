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
# 1. Load Cached Resources
# ═══════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    return json.loads(Path("employee_schema.json").read_text())

@st.cache_data
def load_stats():
    return json.loads(Path("numeric_stats.json").read_text())

@st.cache_data
def load_tooltips():
    try:
        return json.loads(Path("feature_tooltips.json").read_text())
    except:
        return {}

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ═══════════════════════════════════════
# 2. Initialization
# ═══════════════════════════════════════
model       = load_model()
schema_meta = load_schema()
num_stats   = load_stats()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ═══════════════════════════════════════
# 3. Helpers
# ═══════════════════════════════════════
def label_risk(p):
    if p < 0.30: return "🟢 Low"
    elif p < 0.60: return "🟡 Moderate"
    else: return "🔴 High"

def safe_stats(col):
    if col in num_stats:
        cmin, cmax = num_stats[col]["min"], num_stats[col]["max"]
        cmean = num_stats[col]["mean"]
        if cmin == cmax:
            return cmin, cmin + 1, cmin
        return cmin, cmax, cmean
    return 0.0, 1.0, 0.5

# ═══════════════════════════════════════
# 4. Header
# ═══════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown("""
    - Use the **sidebar** to enter employee attributes.
    - Or click **Use Sample Data** to prefill fields.
    - Or click **Reset Form** to clear fields.
    - Upload CSV for batch predictions.
    - Review predictions, SHAP plots, and download history.
    """)

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ═══════════════════════════════════════
# 5. Sidebar Inputs
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")

        if meta["dtype"] == "object":
            options = meta.get("options", ["Unknown"])
            default = st.session_state.get(key, options[0])
            if default not in options:
                default = options[0]
            row[col] = st.sidebar.selectbox(col, options, index=options.index(default), key=key, help=tip)
        else:
            cmin, cmax, cmean = safe_stats(col)
            val = float(st.session_state.get(key, cmean))
            val = min(max(val, cmin), cmax)
            if cmax - cmin <= 1e-9:
                row[col] = st.sidebar.number_input(col, value=val, key=key, help=tip)
            else:
                row[col] = st.sidebar.slider(col, cmin, cmax, val, key=key, help=tip)
    return pd.DataFrame([row])

# ═══════════════════════════════════════
# 6. Sample & Reset Buttons
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
# 7. Load Data (User or Batch)
# ═══════════════════════════════════════
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# ═══════════════════════════════════════
# 8. Preprocess & Predict
# ═══════════════════════════════════════
template = {col: (meta["options"][0] if meta["dtype"] == "object" else 0)
            for col, meta in schema_meta.items()}
X_schema = pd.DataFrame([template])

X_full = pd.concat([raw_df, X_schema], ignore_index=True)
X_enc = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

if batch_mode:
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    raw_df["Prediction"] = np.where(preds == 1, "Yes", "No")
    raw_df["Probability"] = (probs * 100).round(1).astype(str) + " %"
    raw_df["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    row_sel = st.selectbox("Select row to inspect", range(len(raw_df)), format_func=lambda x: f"Row {x+1}")
    st.dataframe(raw_df)
    X_user = X_enc.iloc[[row_sel]]
    user_row = raw_df.iloc[[row_sel]]
else:
    X_user = X_enc.iloc[[0]]
    user_row = raw_df.iloc[[0]]

# ═══════════════════════════════════════
# 9. Single Prediction
# ═══════════════════════════════════════
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk = label_risk(prob)

st.subheader("🎯 Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk)

# ═══════════════════════════════════════
# 10. SHAP Explanation
# ═══════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
shap_vals = explainer.shap_values(X_user)
if isinstance(shap_vals, list): shap_vals = shap_vals[1]

st.markdown("### 🧬 Global Impact — Beeswarm")
f1, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(f1); plt.clf()

st.markdown("### 🧭 Decision Path")
f2, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(f2); plt.clf()

st.markdown("### 🔎 Interactive Feature Impact")
feature = st.selectbox("Choose Feature", X_user.columns)
f3, _ = plt.subplots()
shap.bar_plot(shap_vals[0][X_user.columns.get_loc(feature)], feature_names=[feature], show=False)
st.pyplot(f3); plt.clf()

# ═══════════════════════════════════════
# 11. Prediction History
# ═══════════════════════════════════════
user_row["Prediction"] = "Yes" if pred else "No"
user_row["Probability"] = f"{prob:.1%}"
user_row["Risk Category"] = risk
user_row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat([st.session_state["history"], user_row], ignore_index=True)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist, "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
