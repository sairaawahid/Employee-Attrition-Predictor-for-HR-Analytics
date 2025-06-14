import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1.  CACHED LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    return pd.read_json("employee_schema.json")

@st.cache_data
def load_stats():
    return json.loads(Path("numeric_stats.json").read_text())

@st.cache_data
def load_tooltips():
    return json.loads(Path("feature_tooltips.json").read_text())

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model            = load_model()
schema           = load_schema()
X_stats          = load_stats()
feature_tooltips = load_tooltips()
explainer        = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE HEADER
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar **or** upload a CSV file.  
        2. View predicted **attrition risk** and the color-coded risk card.  
        3. Select an employee row (if CSV) to inspect individual SHAP plots.  
        4. Use these insights to design proactive HR interventions.
        """
    )

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR INPUTS (SINGLE EMPLOYEE)
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        base = col.split("_")[0]          # match tooltip for one-hot fields
        tip  = feature_tooltips.get(base, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique(), help=tip)
        else:
            if col in X_stats:
                cmin, cmax, cmean = (
                    float(X_stats[col]["min"]),
                    float(X_stats[col]["max"]),
                    float(X_stats[col]["mean"]),
                )
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = (
                st.sidebar.number_input(col, value=cmin, help=tip)
                if cmin == cmax
                else st.sidebar.slider(col, cmin, cmax, cmean, help=tip)
            )
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 5.  CSV UPLOAD OR MANUAL INPUT
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload Employee CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(raw_df)} employees from CSV.")
else:
    raw_df = user_input_features()

# One-hot encode & align with training schema
X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_pred = X_enc.iloc[: len(raw_df)]

# ──────────────────────────────────────────────────────────────
# 6.  PREDICTION & RISK CARD(S)
# ──────────────────────────────────────────────────────────────
preds = model.predict(X_pred)
probs = model.predict_proba(X_pred)[:, 1]

if len(raw_df) == 1:
    pred, prob = preds[0], probs[0]

    # Risk category
    risk_label = (
        "🟢 Low"      if prob < 0.30 else
        "🟡 Moderate" if prob < 0.60 else
        "🔴 High"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", "Yes" if pred else "No")
    col2.metric("Attrition Probability", f"{prob:.1%}")
    col3.metric("Risk Category", risk_label)

else:
    results = raw_df.copy()
    results["Prediction"]  = np.where(preds == 1, "Yes", "No")
    results["Probability"] = (probs * 100).round(1).astype(str) + " %"
    results["Risk"] = pd.cut(
        probs, bins=[0, 0.30, 0.60, 1.00],
        labels=["Low", "Moderate", "High"]
    )
    st.subheader("📑 Batch Predictions")
    st.dataframe(results)

# ──────────────────────────────────────────────────────────────
# 7.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

# Row selector (only shown for batch mode)
row_idx = 0
if len(raw_df) > 1:
    row_idx = st.number_input(
        "Select employee row for detailed explanation:",
        min_value=0, max_value=len(raw_df) - 1, step=1, value=0
    )

X_row       = X_pred.iloc[[row_idx]]
shap_values = explainer.shap_values(X_row)
if not isinstance(shap_values, np.ndarray):  # binary model returns list
    shap_values = shap_values[1]

# 7-A Global Beeswarm (for selected row input)
st.markdown("### 🌐 Global Impact — Beeswarm")
fig_bee, _ = plt.subplots()
shap.summary_plot(shap_values, X_row, show=False)
st.pyplot(fig_bee)
plt.clf()

# 7-B Decision Plot
st.markdown("### 🧭 Decision Path (Individual)")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_values[0], X_row, show=False)
st.pyplot(fig_dec)
plt.clf()

# 7-C Force Plot
st.markdown("### 🎯 Local Force Plot")
fig_force = shap.plots.force(
    explainer.expected_value,
    shap_values[0],
    X_row.iloc[0],
    matplotlib=True,
    show=False
)
st.pyplot(fig_force)

st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
