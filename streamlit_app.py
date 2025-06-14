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
model       = load_model()
schema      = load_schema()
X_stats     = load_stats()
tooltips    = load_tooltips()
explainer   = get_explainer(model)

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
        tooltip = tooltips.get(col, "")
        label = f"{col} ℹ️" if tooltip else col

        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(label, schema[col].unique(), help=tooltip)
        else:
            if col in X_stats:
                cmin  = float(X_stats[col]["min"])
                cmax  = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            data[col] = (
                st.sidebar.number_input(label, value=cmin, help=tooltip)
                if cmin == cmax
                else st.sidebar.slider(label, cmin, cmax, cmean, help=tooltip)
            )
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 5.  MAIN WORKFLOW (SINGLE / BATCH MODE)
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Or upload a CSV for batch predictions", type=["csv"])
input_df = None

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    X_full = pd.concat([uploaded_df, schema]).drop_duplicates(keep="first")
    X_enc = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    input_df = X_enc.iloc[:len(uploaded_df)]
    st.success(f"{len(input_df)} records loaded from uploaded CSV.")
else:
    input_df = user_input_features()

# ──────────────────────────────────────────────────────────────
# 5-A  PREDICT
# ──────────────────────────────────────────────────────────────
preds = model.predict(input_df)
probs = model.predict_proba(input_df)[:, 1]

if uploaded_file:
    st.subheader("📊 Batch Prediction Results")
    results = uploaded_df.copy()
    results["Attrition Risk"] = np.where(preds == 1, "Yes", "No")
    results["Probability"] = [f"{p:.1%}" for p in probs]
    st.dataframe(results)

    st.subheader("🔍 SHAP Global Explanation (All Uploaded Records)")
    shap_vals_all = explainer.shap_values(input_df)
    shap_summary = shap_vals_all if isinstance(shap_vals_all, np.ndarray) else shap_vals_all[1]
    fig_bee, ax_bee = plt.subplots()
    shap.summary_plot(shap_summary, input_df, show=False)
    st.pyplot(fig_bee)
    plt.clf()
else:
    pred = preds[0]
    prob = probs[0]

    st.subheader("Prediction")
    st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
    st.write(f"**Probability:** {prob:.1%}")

    # ──────────────────────────────────────────────────────────────
    # 5-B  DASHBOARD METRIC CARD
    # ──────────────────────────────────────────────────────────────
    if prob < 0.3:
        risk_label = "🟢 Low Risk"
        risk_color = "green"
    elif 0.3 <= prob < 0.6:
        risk_label = "🟡 Moderate Risk"
        risk_color = "orange"
    else:
        risk_label = "🔴 High Risk"
        risk_color = "red"

    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <div style="background-color: #f9f9f9; padding: 25px 40px; border-radius: 12px;
                    box-shadow: 2px 2px 12px #ccc; text-align: center; width: 350px;">
            <h4 style="color: #333;">📊 Risk Dashboard</h4>
            <h2 style="color: {risk_color}; margin: 10px 0;">{risk_label}</h2>
            <p style="font-size: 18px;">Predicted Probability: <strong>{prob:.1%}</strong></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # 6.  SHAP CALCULATION
    # ──────────────────────────────────────────────────────────────
    shap_vals = explainer.shap_values(input_df)
    shap_ex = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]

    # ──────────────────────────────────────────────────────────────
    # 7.  SHAP VISUALS
    # ──────────────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Explanations")

    # 7-A  Global Beeswarm Plot
    st.markdown("### 🌐 Global Impact — Beeswarm")
    fig_bee, ax_bee = plt.subplots()
    shap.summary_plot(shap_ex, input_df, show=False)
    st.pyplot(fig_bee)
    plt.clf()

    # 7-B  Individual Decision Plot
    st.markdown("### 🧭 Decision Path (Individual)")
    fig_dec, ax_dec = plt.subplots()
    shap.decision_plot(
        explainer.expected_value,
        shap_ex[0],
        input_df,
        show=False
    )
    st.pyplot(fig_dec)
    plt.clf()

    # 7-C  Individual Force Plot
    st.markdown("### 🎯 Local Force Plot")
    fig = shap.plots.force(
        explainer.expected_value,
        shap_ex[0],
        input_df.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)

    st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
