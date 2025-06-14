import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1.  CACHE HELPERS
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
# 2.  INIT
# ──────────────────────────────────────────────────────────────
model       = load_model()
schema      = load_schema()
X_stats     = load_stats()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE HEADER
# ──────────────────────────────────────────────────────────────
st.title("🧠 Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore **SHAP** explanations for individuals or CSV batches.")

with st.expander("📘 How to use"):
    st.markdown("""
    * Use the sidebar to enter details for a **single employee**, **or** upload a CSV to score multiple employees.  
    * The app shows attrition predictions and interactive SHAP explanations.  
    * Use the row selector (when a CSV is uploaded) to inspect individual employees.
    """)

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR FORM (single employee)
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Input Employee Data")

def single_employee_form() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        tip = tooltips.get(col, "")
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
# 5.  CSV UPLOAD
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV for Batch Prediction", type=["csv"])

if uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
    st.subheader("📑 Uploaded Data Preview")
    st.dataframe(df_csv.head())

    # One–hot encode & align columns
    X_csv = pd.get_dummies(df_csv).reindex(columns=schema.columns, fill_value=0)

    # Predictions
    preds = model.predict(X_csv)
    probs = model.predict_proba(X_csv)[:, 1]

    results = df_csv.copy()
    results["Attrition Prediction"] = np.where(preds == 1, "Yes", "No")
    results["Probability"] = (probs * 100).round(1).astype(str) + "%"

    st.subheader("🔮 Batch Predictions")
    st.dataframe(results)

    # SHAP values for batch
    shap_vals_full = explainer.shap_values(X_csv)
    shap_vals_full = shap_vals_full if isinstance(shap_vals_full, np.ndarray) else shap_vals_full[1]

    st.subheader("🌐 SHAP Beeswarm (All Employees)")
    fig_bee, _ = plt.subplots()
    shap.summary_plot(shap_vals_full, X_csv, show=False)
    st.pyplot(fig_bee)
    plt.clf()

    # Row selector for individual inspection
    st.markdown("### 👤 Inspect Individual Employee")
    row_idx = st.number_input(
        "Select row index:", min_value=0, max_value=len(df_csv) - 1, step=1, value=0
    )

    st.write(f"Showing SHAP explanation for employee row **{row_idx}**")

    # Decision plot
    st.markdown("#### 🧭 Decision Path")
    fig_dec, _ = plt.subplots()
    shap.decision_plot(
        explainer.expected_value,
        shap_vals_full[row_idx],
        X_csv.iloc[[row_idx]],
        show=False
    )
    st.pyplot(fig_dec)
    plt.clf()

    # Force plot
    st.markdown("#### 🎯 Local Force Plot")
    fig_force = shap.plots.force(
        explainer.expected_value,
        shap_vals_full[row_idx],
        X_csv.iloc[row_idx],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig_force)

else:
    # ──────────────────────────────────────────────────────────────
    # 6.  SINGLE EMPLOYEE PREDICTION
    # ──────────────────────────────────────────────────────────────
    input_df = single_employee_form()
    X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
    X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_enc.iloc[[0]]

    pred = model.predict(X_user)[0]
    prob = model.predict_proba(X_user)[0, 1]

    st.subheader("🔮 Prediction")
    st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
    st.write(f"**Probability:** {prob:.1%}")

    # SHAP single
    shap_vals_single = explainer.shap_values(X_user)
    shap_vals_single = shap_vals_single if isinstance(shap_vals_single, np.ndarray) else shap_vals_single[1]

    st.subheader("🔍 SHAP Explanations")

    st.markdown("#### 🌐 Global Impact — Beeswarm")
    fig_bee, _ = plt.subplots()
    shap.summary_plot(shap_vals_single, X_user, show=False)
    st.pyplot(fig_bee)
    plt.clf()

    st.markdown("#### 🧭 Decision Path")
    fig_dec, _ = plt.subplots()
    shap.decision_plot(
        explainer.expected_value,
        shap_vals_single[0],
        X_user,
        show=False
    )
    st.pyplot(fig_dec)
    plt.clf()

    st.markdown("#### 🎯 Local Force Plot")
    fig_force = shap.plots.force(
        explainer.expected_value,
        shap_vals_single[0],
        X_user.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig_force)

    st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
