import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
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
    """schema_meta: {col: {dtype: ..., [options|min|max|mean]}}"""
    return json.loads(Path("employee_schema.json").read_text())

@st.cache_data
def load_stats():
    fp = Path("numeric_stats.json")
    return json.loads(fp.read_text()) if fp.exists() else {}

@st.cache_data
def load_tooltips():
    try:
        return json.loads(Path("feature_tooltips.json").read_text())
    except Exception:
        return {}

# ••• leading “_” so the model isn't hashed (fixes UnhashableParamError)
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ═════════════════════════════════════════════════════════════
# 2.  INITIALISE
# ═════════════════════════════════════════════════════════════
model         = load_model()
model_features = model.get_booster().feature_names  # ← NEW
schema_meta   = load_schema()
num_stats     = load_stats()
tooltips      = load_tooltips()
explainer     = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ═════════════════════════════════════════════════════════════
# 3.  HELPERS
# ═════════════════════════════════════════════════════════════
def label_risk(p):
    if p < .30: return "🟢 Low"
    if p < .60: return "🟡 Moderate"
    return "🔴 High"

def safe_stats(col):
    if col in num_stats:
        cmin, cmax = num_stats[col]["min"], num_stats[col]["max"]
        cmean      = num_stats[col]["mean"]
        if cmin == cmax:              # constant → widen to avoid slider crash
            return cmin, cmax + 1, cmin
        return cmin, cmax, cmean
    return 0.0, 1.0, 0.5

# ═════════════════════════════════════════════════════════════
# 4.  UI HEADER
# ═════════════════════════════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        * **Sidebar**: enter attributes — or click **Use Sample Data** / **Reset Form**.  
        * Optional: **Upload CSV** for batch scoring.  
        * View **prediction, SHAP charts, interactive feature impact**, and downloadable history.
        """
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ═════════════════════════════════════════════════════════════
# 5.  SIDEBAR INPUTS
# ═════════════════════════════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        tip = tooltips.get(col.split("_")[0], "")
        key = f"inp_{col}"

        if meta["dtype"] == "object":
            opts = meta.get("options", ["Unknown"])
            default = st.session_state.get(key, opts[0])
            if default not in opts:
                default = opts[0]
            row[col] = st.sidebar.selectbox(
                col, opts, index=opts.index(default), key=key, help=tip
            )
        else:
            cmin, cmax, cmean = safe_stats(col)
            default = float(st.session_state.get(key, cmean))
            default = max(min(default, cmax), cmin)
            if cmax - cmin <= 1e-9:
                row[col] = st.sidebar.number_input(
                    col, value=default, key=key, help=tip
                )
            else:
                row[col] = st.sidebar.slider(
                    col, cmin, cmax, default, key=key, help=tip
                )
    return pd.DataFrame(row, index=[0])

# ═════════════════════════════════════════════════════════════
# 6.  SAMPLE & RESET BUTTONS
# ═════════════════════════════════════════════════════════════
sample_employee = {
    "Age": 32,
    "Gender": "Male",
    "Department": "Research & Development",
    "BusinessTravel": "Travel_Rarely",
    "MonthlyIncome": 5200,
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
st.sidebar.button("🔄 Reset Form",     on_click=reset_form)

# ═════════════════════════════════════════════════════════════
# 7.  COLLECT INPUT ROW(S)
# ═════════════════════════════════════════════════════════════
if uploaded_file is not None:
    raw_df   = pd.read_csv(uploaded_file)
    batch_on = True
else:
    raw_df   = sidebar_inputs()
    batch_on = False

# ═════════════════════════════════════════════════════════════
# 8.  PREPARE FOR MODEL
# ═════════════════════════════════════════════════════════════
# build a template row so every expected category exists:
template = {col: (meta["options"][0] if meta["dtype"] == "object" else 0)
            for col, meta in schema_meta.items()}
schema_df = pd.DataFrame([template])

X_full  = pd.concat([raw_df, schema_df], ignore_index=True)
X_dmy   = pd.get_dummies(X_full)

# ✔ align EXACTLY to model's training features
X_enc   = X_dmy.reindex(columns=model_features, fill_value=0).iloc[: len(raw_df)]
X_user  = X_enc.iloc[[0]]

# ═════════════════════════════════════════════════════════════
# 9.  PREDICT
# ═════════════════════════════════════════════════════════════
pred  = model.predict(X_user)[0]
prob  = model.predict_proba(X_user)[0, 1]
risk  = label_risk(prob)

st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction",      "Yes" if pred else "No")
c2.metric("Probability",     f"{prob:.1%}")
c3.metric("Risk Category",   risk)

# ═════════════════════════════════════════════════════════════
# 10.  SHAP VISUALS
# ═════════════════════════════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
sv = explainer.shap_values(X_user)
if isinstance(sv, list):
    sv = sv[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
f1, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(f1); plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
f2, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(f2); plt.clf()

st.markdown("### 🎯 Local Force Plot")
try:
    f3 = shap.plots.force(explainer.expected_value, sv[0], X_user.iloc[0],
                          matplotlib=True, show=False)
    st.pyplot(f3)
except Exception:
    st.info("Force plot fallback to waterfall.")
    f3, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=sv[0],
                         base_values=explainer.expected_value,
                         data=X_user.iloc[0]),
        max_display=15, show=False)
    st.pyplot(f3)

# ═════════════════════════════════════════════════════════════
# 11.  BATCH SUMMARY
# ═════════════════════════════════════════════════════════════
if batch_on:
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    out   = raw_df.copy()
    out["Prediction"]    = np.where(preds == 1, "Yes", "No")
    out["Probability"]   = (probs * 100).round(1).astype(str) + " %"
    out["Risk Category"] = [label_risk(p) for p in probs]
    st.markdown("### 📑 Batch Prediction Summary")
    st.dataframe(out)

# ═════════════════════════════════════════════════════════════
# 12.  HISTORY
# ═════════════════════════════════════════════════════════════
hist = raw_df.iloc[[0]].copy()
hist["Prediction"]    = "Yes" if pred else "No"
hist["Probability"]   = f"{prob:.1%}"
hist["Risk Category"] = risk
hist["Timestamp"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat(
    [st.session_state["history"], hist], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
