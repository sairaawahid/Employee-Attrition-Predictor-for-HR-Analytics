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

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model = load_model()
schema = load_schema()
X_stats = load_stats()
explainer = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE TITLE / INSTRUCTIONS
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")

with st.expander("📘 How to use this app"):
    st.markdown("""
    - Use the sidebar to input attributes for a single employee.
    - Or upload a CSV file for batch predictions.
    - View attrition predictions and explanation visuals (SHAP).
    """)

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 4.  TOOLTIP HELPER
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_tooltips():
    return json.loads(Path("feature_tooltips.json").read_text())

feature_tooltips = load_tooltips()

# ──────────────────────────────────────────────────────────────
# 5.  USER INPUT FORM (SINGLE EMPLOYEE)
# ──────────────────────────────────────────────────────────────
def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        tooltip = feature_tooltips.get(col, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(f"{col}", schema[col].unique(), help=tooltip)
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

# ──────────────────────────────────────────────────────────────
# 6.  CSV UPLOAD HANDLING
# ──────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV for Batch Prediction", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.subheader("📑 Batch Prediction Results")

    # Encode uploaded data
    df_encoded = pd.get_dummies(df_uploaded).reindex(columns=schema.columns, fill_value=0)

    # Predict
    preds = model.predict(df_encoded)
    probs = model.predict_proba(df_encoded)[:, 1]

    results = df_uploaded.copy()
    results["Attrition Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    results["Attrition Probability"] = [f"{pr:.1%}" for pr in probs]

    st.dataframe(results)

    # Download button
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", data=csv, file_name="batch_attrition_predictions.csv", mime="text/csv")

else:
    # Run single employee prediction only when no file is uploaded
    input_df = user_input_features()

    # Encode & predict
    X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
    X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_enc.iloc[[0]]

    pred = model.predict(X_user)[0]
    prob = model.predict_proba(X_user)[0, 1]

    st.subheader("Prediction")
    st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
    st.write(f"**Probability:** {prob:.1%}")

    # SHAP calculations
    raw_shap = explainer.shap_values(X_user)
    shap_vals = raw_shap if isinstance(raw_shap, np.ndarray) else raw_shap[1]

    st.subheader("🔍 SHAP Explanations")

    st.markdown("### 1. Global Impact — Beeswarm")
    fig_bee, _ = plt.subplots()
    shap.summary_plot(shap_vals, X_user, show=False)
    st.pyplot(fig_bee)
    plt.clf()

    st.markdown("### 2. Decision Path (Individual)")
    fig_dec, _ = plt.subplots()
    shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
    st.pyplot(fig_dec)
    plt.clf()

    st.markdown("### 3. Local Force Plot")
    fig = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig)

    st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
