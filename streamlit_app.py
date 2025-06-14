import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import pathlib

# ------------------------------------------------------------------
# 1️⃣  LOAD ARTIFACTS
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    # first encoded row saved from training set
    return pd.read_json("employee_schema.json")

@st.cache_data
def load_stats():
    # min / max / mean for every numeric column you have stats for
    return json.loads(pathlib.Path("numeric_stats.json").read_text())

model   = load_model()
schema  = load_schema()
X_stats = load_stats()          # dict: {col: {"min": v, "max": v, "mean": v}}

# ------------------------------------------------------------------
# 2️⃣  PAGE TITLE
# ------------------------------------------------------------------

st.title("🧠 Employee Attrition Predictor")
st.markdown(
    """
    Predict an employee’s attrition risk and see which factors matter most.
    Model: **XGBoost** • Interpretation: **SHAP**
    """
)

# ------------------------------------------------------------------
# 3️⃣  SIDEBAR INPUT UI
# ------------------------------------------------------------------

st.sidebar.header("Employee Attributes")

def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:

        # ------------ CATEGORICAL -------------
        if schema[col].dtype == "object":
            options = schema[col].unique()
            data[col] = st.sidebar.selectbox(col, options)

        # ------------ NUMERIC -----------------
        else:
            if col in X_stats:                # stats available
                cmin  = float(X_stats[col]["min"])
                cmax  = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:                             # fallback for missing stats
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            # Streamlit slider fails if min == max
            if cmin == cmax:
                data[col] = st.sidebar.number_input(col, value=cmin)
            else:
                data[col] = st.sidebar.slider(col, cmin, cmax, cmean)

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ------------------------------------------------------------------
# 4️⃣  ENCODE USER INPUT TO MATCH MODEL
# ------------------------------------------------------------------

# concatenate then one-hot so every categorical column appears
X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full)

# re-index to schema column order; new unseen cols default to 0
X_enc  = X_enc.reindex(columns=schema.columns, fill_value=0)

# only the first row (the user input) is needed for prediction
X_user = X_enc.iloc[[0]]

# ------------------------------------------------------------------
# 5️⃣  PREDICT
# ------------------------------------------------------------------

pred     = model.predict(X_user)[0]
prob     = model.predict_proba(X_user)[0][1]

st.subheader("Prediction")
st.write(f"**Attrition Risk:** {'Yes' if pred == 1 else 'No'}")
st.write(f"**Probability:** {prob:.1%}")

# ------------------------------------------------------------------
# 6️⃣  SHAP EXPLANATION  (bar plot for current input)
# ------------------------------------------------------------------

st.subheader("Feature Contribution (SHAP)")

# cache explainer so it's created once
@st.cache_resource
def get_explainer(m):
    return shap.TreeExplainer(m)

explainer   = get_explainer(model)
shap_values = explainer.shap_values(X_user)

st.set_option("deprecation.showPyplotGlobalUse", False)
plt.title("Top Features Driving This Prediction")
shap.summary_plot(shap_values, X_user, plot_type="bar", show=False)
st.pyplot(bbox_inches="tight")

st.caption("Positive SHAP values push toward leaving; negative push toward staying.")
