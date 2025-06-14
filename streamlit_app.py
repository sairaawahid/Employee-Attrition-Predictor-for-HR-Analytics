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
model     = load_model()
schema    = load_schema()
X_stats   = load_stats()
tooltips  = load_tooltips()
explainer = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE TITLE / USER HELP
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **XGBoost + SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown("""
    1. **Enter employee details** in the sidebar or **upload a CSV file**.
    2. View the predicted **attrition risk** and **feature explanations**.
    3. Use the results to guide HR decision-making and proactive interventions.
    """)

# ──────────────────────────────────────────────────────────────
# 4.  INPUT SECTION
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Input Employee Data")

def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        label = f"{col}"
        tooltip = tooltips.get(col, "")
        help_text = tooltip if tooltip else None

        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(label, schema[col].unique(), help=help_text)
        else:
            if col in X_stats:
                cmin = float(X_stats[col]["min"])
                cmax = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            data[col] = (
                st.sidebar.number_input(label, value=cmin, help=help_text)
                if cmin == cmax
                else st.sidebar.slider(label, cmin, cmax, cmean, help=help_text)
            )
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 5.  BATCH UPLOAD
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV for Batch Prediction", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(uploaded_df)

        # One-hot encode and align with training schema
        X_uploaded_enc = pd.get_dummies(uploaded_df)
        X_uploaded_enc = X_uploaded_enc.reindex(columns=schema.columns, fill_value=0)

        # Predict
        preds = model.predict(X_uploaded_enc)
        probs = model.predict_proba(X_uploaded_enc)[:, 1]
        uploaded_df["Attrition Risk"] = ["Yes" if p == 1 else "No" for p in preds]
        uploaded_df["Probability"] = [f"{p:.1%}" for p in probs]

        st.subheader("📊 Predictions")
        st.dataframe(uploaded_df[["Attrition Risk", "Probability"]])

        # SHAP Summary Plot for uploaded data
        st.subheader("🔍 SHAP Explanation — Uploaded Data")
        shap_vals = explainer.shap_values(X_uploaded_enc)
        shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]
        fig_uploaded, ax_uploaded = plt.subplots()
        shap.summary_plot(shap_vals, X_uploaded_enc, show=False)
        st.pyplot(fig_uploaded)
        plt.clf()

    except Exception as e:
        st.error(f"Error processing uploaded CSV: {e}")
        st.stop()

else:
    # ──────────────────────────────────────────────────────────────
    # 6.  INDIVIDUAL INPUT PROCESSING
    # ──────────────────────────────────────────────────────────────
    input_df = user_input_features()
    X_full   = pd.concat([input_df, schema]).drop_duplicates(keep="first")
    X_enc    = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user   = X_enc.iloc[[0]]

    pred = model.predict(X_user)[0]
    prob = model.predict_proba(X_user)[0, 1]

    st.subheader("🎯 Prediction")
    st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
    st.write(f"**Probability:** {prob:.1%}")

    # SHAP Calculation
    raw_shap  = explainer.shap_values(X_user)
    shap_vals = raw_shap if isinstance(raw_shap, np.ndarray) else raw_shap[1]

    st.subheader("🔍 SHAP Explanations")

    # A. Beeswarm
    st.markdown("### 🌐 Global Impact — Beeswarm")
    fig_bee, ax_bee = plt.subplots()
    shap.summary_plot(shap_vals, X_user, show=False)
    st.pyplot(fig_bee)
    plt.clf()

    # B. Decision Path
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

    # C. Force Plot (static)
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
