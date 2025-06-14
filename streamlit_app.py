import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and example schema
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_optimized_model.pkl')
    return model

@st.cache_data
def load_schema():
    return pd.read_json("employee_schema.json")

model = load_model()
schema = load_schema()

st.title("🧠 Employee Attrition Predictor")
st.markdown("Predict the risk of attrition based on employee attributes. Powered by XGBoost and SHAP.")

# Sidebar input
st.sidebar.header("Input Employee Data")

def user_input_features():
    data = {}
    for col in schema.columns:
    if schema[col].dtype == "object":
        # categorical
        data[col] = st.sidebar.selectbox(col, schema[col].unique())
    else:
        cmin = float(X_stats[col]["min"])   # see stats dict below
        cmax = float(X_stats[col]["max"])
        if cmin == cmax:                   # 👉 identical values → use number_input
            data[col] = st.sidebar.number_input(col, value=cmin)
        else:
            data[col] = st.sidebar.slider(col, cmin, cmax, float(X_stats[col]["mean"]))
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Encode user input to match model (same columns)
X = pd.concat([input_df, schema]).drop_duplicates(keep='first')  # ensure all columns
X_encoded = pd.get_dummies(X)
X_encoded = X_encoded.reindex(columns=schema.columns, fill_value=0)

# Predict
prediction = model.predict(X_encoded)[0]
probability = model.predict_proba(X_encoded)[0][1]

st.subheader("Prediction:")
st.write(f"**Attrition Risk: {'Yes' if prediction == 1 else 'No'}**")
st.write(f"**Probability: {probability:.2%}**")

# SHAP Explanation
st.subheader("Explanation (SHAP Feature Impact):")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_encoded)

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.title("Feature Impact (SHAP)")
shap.summary_plot(shap_values, X_encoded, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')
