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
model       = load_model()
schema      = load_schema()
X_stats     = load_stats()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  TITLE & USER GUIDANCE
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar — or upload a CSV.  
        2. The app predicts **attrition risk** for one or more employees.  
        3. **SHAP plots** help visualize feature impact.  
        4. Use this tool to identify and support at-risk employees proactively.
        """
    )

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 4.  USER INPUT SECTION (SIDEBAR)
# ──────────────────────────────────────────────────────────────
def user_input_features():
    data = {}
    for col in schema.columns:
        tooltip = tooltips.get(col, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(f"{col}", schema[col].unique(), help=tooltip)
        else:
            if col in X_stats:
                cmin = float(X_stats[col]["min"])
                cmax = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            if cmin == cmax:
                data[col] = st.sidebar.number_input(f"{col}", value=cmin, help=tooltip)
            else:
                data[col] = st.sidebar.slider(f"{col}", cmin, cmax, cmean, help=tooltip)
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 5.  BATCH UPLOAD (CSV)
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Or Upload CSV for Batch Prediction", type=["csv"])
if uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview", df_csv.head())

    # Ensure all expected columns are there
    missing_cols = [col for col in schema.columns if col not in df_csv.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from uploaded CSV: {missing_cols}")
    else:
        X_full = pd.concat([df_csv, schema]).drop_duplicates(keep="first")
        X_enc = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
        X_user = X_enc.iloc[:len(df_csv)]

        preds = model.predict(X_user)
        probs = model.predict_proba(X_user)[:, 1]

        st.subheader("Batch Predictions")
        results = df_csv.copy()
        results["Attrition Prediction"] = np.where(preds == 1, "Yes", "No")
        results["Attrition Probability"] = (probs * 100).round(1).astype(str) + '%'
        st.dataframe(results)

        st.subheader("SHAP Explanation (Top 3 Examples)")
        shap_vals = explainer.shap_values(X_user)
        for i in range(min(3, len(X_user))):
            st.markdown(f"#### Employee {i+1}")
            fig = shap.plots.force(
                explainer.expected_value,
                shap_vals[i],
                X_user.iloc[i],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)

else:
    # ──────────────────────────────────────────────────────────────
    # 6.  SINGLE PREDICTION FROM SIDEBAR
    # ──────────────────────────────────────────────────────────────
    input_df = user_input_features()
    X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
    X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_enc.iloc[[0]]

    pred = model.predict(X_user)[0]
    prob = model.predict_proba(X_user)[0, 1]

    st.subheader("Prediction")
    st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
    st.write(f"**Probability:** {prob:.1%}")

    # ──────────────────────────────────────────────────────────────
    # 7.  SHAP SINGLE EMPLOYEE PLOTS
    # ──────────────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Explanations")

    raw_shap = explainer.shap_values(X_user)
    shap_vals = raw_shap if isinstance(raw_shap, np.ndarray) else raw_shap[1]

    # 7-A Beeswarm
    st.markdown("### 🌐 Global Impact — Beeswarm")
    fig_bee, ax_bee = plt.subplots()
    shap.summary_plot(shap_vals, X_user, show=False)
    st.pyplot(fig_bee)
    plt.clf()

    # 7-B Decision Plot
    st.markdown("### 🧭 Decision Path (Individual)")
    fig_dec, ax_dec = plt.subplots()
    shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
    st.pyplot(fig_dec)
    plt.clf()

    # 7-C Force Plot (matplotlib safe)
    st.markdown("### 🎯 Local Force Plot")
    fig = shap.plots.force(
        explainer.expected_value,
        shap_vals[0],
        X_user.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
