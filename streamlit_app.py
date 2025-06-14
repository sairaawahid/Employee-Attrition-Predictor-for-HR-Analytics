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
    """
    Loads feature descriptions from feature_tooltips.json.
    If the file is missing, returns an empty dict.
    """
    p = Path("feature_tooltips.json")
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model      = load_model()
schema     = load_schema()
X_stats    = load_stats()
tooltips   = load_tooltips()
explainer  = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE HEADER & HELP
# ──────────────────────────────────────────────────────────────
st.title("🧠 Employee Attrition Predictor")
st.markdown(
    "Predict attrition risk and explore model explanations with **SHAP**."
)

with st.expander("📘 How to use this app", expanded=False):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar (or leave defaults).  
        2. The main panel updates with **attrition risk & probability**.  
        3. Scroll to **SHAP charts** to see which features drive the prediction.  
        4. Use these insights to design targeted HR interventions.
        """
    )

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR INPUTS WITH TOOLTIPS
# ──────────────────────────────────────────────────────────────
def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        tip = tooltips.get(col, "")  # tooltip text (blank if not found)

        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(
                label=col,
                options=schema[col].unique(),
                help=tip
            )
        else:
            # numeric columns
            if col in X_stats:
                cmin, cmax, cmean = (
                    float(X_stats[col]["min"]),
                    float(X_stats[col]["max"]),
                    float(X_stats[col]["mean"]),
                )
            else:  # fallback
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            data[col] = (
                st.sidebar.number_input(col, value=cmin, help=tip)
                if cmin == cmax
                else st.sidebar.slider(col, cmin, cmax, cmean, help=tip)
            )
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ──────────────────────────────────────────────────────────────
# 5.  ENCODE INPUT & PREDICT
# ──────────────────────────────────────────────────────────────
X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_user = X_enc.iloc[[0]]

pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]

st.subheader("🔮 Prediction")
st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
st.write(f"**Probability:** {prob:.1%}")

# ──────────────────────────────────────────────────────────────
# 6.  SHAP CALCULATION
# ──────────────────────────────────────────────────────────────
raw_shap  = explainer.shap_values(X_user)
shap_vals = raw_shap if isinstance(raw_shap, np.ndarray) else raw_shap[1]

# ──────────────────────────────────────────────────────────────
# 7.  SHAP VISUALS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

# 7-A Beeswarm (global)
st.markdown("### 1. Global Impact — Beeswarm")
fig_bee, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig_bee)
plt.clf()

# 7-B Decision plot (local)
st.markdown("### 2. Decision Path (Individual)")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(fig_dec)
plt.clf()

# 7-C Force plot (local, static)
st.markdown("### 3. Local Force Plot")
fig_force = shap.plots.force(
    explainer.expected_value,
    shap_vals[0],
    X_user.iloc[0],
    matplotlib=True,
    show=False
)
st.pyplot(fig_force)

st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")
