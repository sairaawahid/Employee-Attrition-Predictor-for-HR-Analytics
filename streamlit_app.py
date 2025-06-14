import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Cache model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")


# Cache schema
@st.cache_data
def load_schema():
    return pd.read_json("employee_schema.json")


# Cache numeric stats
@st.cache_data
def load_stats():
    return json.loads(Path("numeric_stats.json").read_text())


# Load model, schema, and stats
model = load_model()
schema = load_schema()
X_stats = load_stats()

# Title
st.title("🧠 Employee Attrition Predictor")
st.markdown("Predict the risk of attrition based on employee attributes. Powered by XGBoost and SHAP.")

st.sidebar.header("Input Employee Data")


# Sidebar input form
def user_input_features():
    data = {}
    for col in schema.columns:
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(f"{col}", schema[col].unique())
        else:
            try:
                cmin = float(X_stats[col]["min"])
                cmax = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            except KeyError:
                cmin, cmax, cmean = 0.0, 1.0, 0.5

            if cmin == cmax:
                data[col] = st.sidebar.number_input(f"{col}", value=cmin)
            else:
                data[col] = st.sidebar.slider(f"{col}", cmin, cmax, cmean)

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# One-hot encode to match training schema
X = pd.concat([input_df, schema]).drop_duplicates(keep="first")
X_encoded = pd.get_dummies(X)
X_encoded = X_encoded.reindex(columns=schema.columns, fill_value=0)

# Predict
prediction = model.predict(X_encoded)[0]
probability = model.predict_proba(X_encoded)[0][1]

st.subheader("Prediction:")
st.write(f"**Attrition Risk: {'Yes' if prediction == 1 else 'No'}**")
st.write(f"**Probability: {probability:.2%}**")


# SHAP Explanation
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = get_explainer(model)
shap_values = explainer.shap_values(X_encoded)

st.subheader("Feature Contribution (SHAP)")

# Plot SHAP summary as bar chart
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_encoded, plot_type="bar", show=False)
st.pyplot(fig)
plt.clf()
