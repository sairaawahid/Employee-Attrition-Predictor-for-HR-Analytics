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

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model   = load_model()
schema  = load_schema()
X_stats = load_stats()
explainer = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE TITLE / SIDEBAR
# ──────────────────────────────────────────────────────────────
st.title("🧠 Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

# Collapsible help — starts open the first time; collapse again if you like
with st.expander("📘 How to use this app", expanded=False):
    st.markdown(
        """
        **Step-by-step**

        1. Select or edit employee details in the sidebar.  
        2. The main panel instantly updates with attrition risk and probability.  
        3. Scroll down to read SHAP charts (global and individual explanations).  
        4. Use the insights to design targeted HR interventions.
        """
    )

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR INPUTS
# ──────────────────────────────────────────────────────────────
def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique())
        else:
            if col in X_stats:
                cmin  = float(X_stats[col]["min"])
                cmax  = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            data[col] = (
                st.sidebar.number_input(col, value=cmin)
                if cmin == cmax
                else st.sidebar.slider(col, cmin, cmax, cmean)
            )
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ──────────────────────────────────────────────────────────────
# 5.  ONE-HOT ENCODE  &  PREDICT
# ──────────────────────────────────────────────────────────────
X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_user = X_enc.iloc[[0]]

pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]

st.subheader("Prediction")
st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
st.write(f"**Probability:** {prob:.1%}")

# ──────────────────────────────────────────────────────────────
# 6.  SHAP CALCULATION
# ──────────────────────────────────────────────────────────────
# For binary model, shap_values may be list [neg_class, pos_class]
raw_shap = explainer.shap_values(X_user)
shap_vals = raw_shap if isinstance(raw_shap, np.ndarray) else raw_shap[1]

# ──────────────────────────────────────────────────────────────
# 7.  SHAP VISUALS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

# 7-A  Global Beeswarm Plot
st.markdown("### 🌐 Global Impact — Beeswarm")
fig_bee, ax_bee = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)   # beeswarm by default
st.pyplot(fig_bee)
plt.clf()

# 7-B  Individual Decision Plot
st.markdown("### 🧭 Decision Path (Individual)")
fig_dec, ax_dec = plt.subplots()
shap.decision_plot(
    explainer.expected_value,
    shap_vals[0],        # decision plot needs 1-D array
    X_user,
    show=False
)
st.pyplot(fig_dec)
plt.clf()

# 7-C  Individual Force Plot (Static matplotlib for Streamlit compatibility)
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
