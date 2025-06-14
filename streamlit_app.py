import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
import streamlit.components.v1 as components

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
model   = load_model()
schema  = load_schema()
X_stats = load_stats()
feature_tooltips = load_tooltips()
explainer = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE TITLE / SIDEBAR
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
       1. **Enter employee details** in the sidebar or **upload a CSV** to score multiple employees.  
        2. The main panel updates with **attrition risk & probability**.  
        3. Scroll to **SHAP charts** to see which features drive the prediction.
        4. Use the row selector (when a CSV is uploaded) to inspect individual employees.  
        5. Use these insights to design targeted HR interventions.
        """
    )

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR INPUTS
# ──────────────────────────────────────────────────────────────
def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        base_col = col.split("_")[0]  # For tooltip matching
        tooltip = feature_tooltips.get(base_col, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique(), help=tooltip)
        else:
            if col in X_stats:
                cmin  = float(X_stats[col]["min"])
                cmax  = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = (
                st.sidebar.number_input(col, value=cmin, help=tooltip)
                if cmin == cmax
                else st.sidebar.slider(col, cmin, cmax, cmean, help=tooltip)
            )
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ──────────────────────────────────────────────────────────────
# 5.  ONE-HOT ENCODE & PREDICT
# ──────────────────────────────────────────────────────────────
X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_user = X_enc.iloc[[0]]

pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]

# ──────────────────────────────────────────────────────────────
# 6.  METRIC CARDS
# ──────────────────────────────────────────────────────────────
st.subheader("📊 Employee Risk Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Prediction", "Yes" if pred else "No")
col2.metric("Risk Probability", f"{prob:.1%}")
col3.metric("Risk Category",
            "🔴 High" if prob > 0.6 else "🟡 Moderate" if prob > 0.3 else "🟢 Low")

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

# ──────────────────────────────────────────────────────────────
# 7.  SHAP CALCULATION
# ──────────────────────────────────────────────────────────────
raw_shap = explainer.shap_values(X_user)
shap_vals = raw_shap if isinstance(raw_shap, np.ndarray) else raw_shap[1]

# ──────────────────────────────────────────────────────────────
# 8.  SHAP VISUALS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

# 8-A Global Beeswarm Plot
st.markdown("### 🌐 Global Impact — Beeswarm")
fig_bee, ax_bee = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig_bee)
plt.clf()

# 8-B Decision Path (Individual)
st.markdown("### 🧭 Decision Path (Individual)")
fig_dec, ax_dec = plt.subplots()
shap.decision_plot(
    explainer.expected_value,
    shap_vals[0],
    X_user,
    show=False
)
st.pyplot(fig_dec)
plt.clf()

# 8-C Force Plot (Static for Streamlit)
st.markdown("### 🎯 Local Force Plot")
fig_force = shap.plots.force(
    explainer.expected_value,
    shap_vals[0],
    X_user.iloc[0],
    matplotlib=True,
    show=False
)
st.pyplot(fig_force)

st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
