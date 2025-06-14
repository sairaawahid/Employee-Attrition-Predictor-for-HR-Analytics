import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import pathlib

# ───────────────────────────────────────────────────────────────────────────────
# 1.  LOAD ARTIFACTS
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    return pd.read_json("employee_schema.json")

@st.cache_data
def load_stats():
    return json.loads(pathlib.Path("numeric_stats.json").read_text())

model   = load_model()
schema  = load_schema()
X_stats = load_stats()

# ───────────────────────────────────────────────────────────────────────────────
st.title("🧠 Employee Attrition Predictor")
st.markdown(
    "Predict attrition risk and see which factors matter most. Powered by **XGBoost** + **SHAP**."
)

# ───────────────────────────────────────────────────────────────────────────────
# 2.  SIDEBAR WIDGETS
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Employee Attributes")

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

# ───────────────────────────────────────────────────────────────────────────────
# 3.  ENCODE INPUT
# ───────────────────────────────────────────────────────────────────────────────
X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_user = X_enc.iloc[[0]]

# ───────────────────────────────────────────────────────────────────────────────
# 4.  PREDICT
# ───────────────────────────────────────────────────────────────────────────────
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]

st.subheader("Prediction")
st.write(f"**Attrition Risk:** {'Yes' if pred else 'No'}")
st.write(f"**Probability:** {prob:.1%}")

# ───────────────────────────────────────────────────────────────────────────────
# 5.  SHAP EXPLANATION
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_explainer(_model):          #   ←  leading underscore fixes hash issue
    return shap.TreeExplainer(_model)

explainer   = get_explainer(model)
shap_values = explainer.shap_values(X_user)

st.subheader("Feature Contribution (SHAP)")
st.set_option("deprecation.showPyplotGlobalUse", False)
plt.title("Top Features Driving This Prediction")
# For binary class, shap returns list; handle both cases
shap.summary_plot(
    shap_values if isinstance(shap_values, np.ndarray) else shap_values[1],
    X_user,
    plot_type="bar",
    show=False,
)
st.pyplot(bbox_inches="tight")
st.caption("Positive SHAP values push toward leaving; negative push toward staying.")
