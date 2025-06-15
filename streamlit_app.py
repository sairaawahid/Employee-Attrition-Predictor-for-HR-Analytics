import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# Load model and data
model = joblib.load("models/rf_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
X_train = joblib.load("models/X_train.pkl")

# Load model zoo for comparison (Feature 9)
model_zoo = {
    "Random Forest": joblib.load("models/rf_model.pkl"),
    "Logistic Regression": joblib.load("models/logreg_model.pkl"),
    "Decision Tree": joblib.load("models/tree_model.pkl"),
}

# Title and UI
st.title("🔍 Employee Attrition Predictor")
st.markdown("Use this app to predict the likelihood of employee attrition based on HR features.")

st.sidebar.header("Input Employee Data")
user_input = {
    "Age": st.sidebar.slider("Age", 18, 60, 30),
    "MonthlyIncome": st.sidebar.slider("Monthly Income", 1000, 20000, 5000),
    "JobSatisfaction": st.sidebar.selectbox("Job Satisfaction (1–4)", [1, 2, 3, 4]),
    "OverTime": st.sidebar.selectbox("OverTime", ["Yes", "No"]),
    "DistanceFromHome": st.sidebar.slider("Distance From Home (km)", 1, 30, 10),
}

input_df = pd.DataFrame([user_input])
input_df["OverTime"] = LabelEncoder().fit(["No", "Yes"]).transform(input_df["OverTime"])

# Risk category helper (Feature 10)
def label_risk(p):
    if p < 0.30:
        return "🟢", "Low"
    elif p < 0.60:
        return "🟡", "Moderate"
    else:
        return "🔴", "High"

# Predict
pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]
risk_emoji, risk_label = label_risk(prob)

# Output
st.subheader("🎯 Prediction Results")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", f"{risk_emoji} {risk_label}")

# SHAP Global Beeswarm
shap_vals = explainer(input_df)
X_user = input_df.copy()
st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig1); plt.clf()

# SHAP Individual Decision Path
st.markdown("### 🧭 Decision Path (Individual)")
fig2, ax = plt.subplots()
shap.plots.bar(shap_vals[0], max_display=10, show=False)
st.pyplot(fig2); plt.clf()

# SHAP Local Force Plot (matplotlib)
st.markdown("### 📌 Local Force Plot")
fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0], matplotlib=True, show=False)
st.pyplot(fig3); plt.clf()

# Model Comparison View (Feature 9)
st.markdown("### 🔍 Model Comparison View")
comparison_data = []
for name, mdl in model_zoo.items():
    p = mdl.predict_proba(input_df)[0][1]
    emoji, label = label_risk(p)
    comparison_data.append({
        "Model": name,
        "Prediction": "Yes" if mdl.predict(input_df)[0] else "No",
        "Probability": f"{p:.1%}",
        "Risk Category": f"{emoji} {label}"
    })
st.dataframe(pd.DataFrame(comparison_data))

# Batch prediction (if file uploaded)
st.markdown("### 📥 Upload Batch CSV")
uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type="csv")
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df = raw_df.copy()
    if "OverTime" in df.columns:
        df["OverTime"] = LabelEncoder().fit(["No", "Yes"]).transform(df["OverTime"])
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    results = raw_df.copy()
    results["Prediction"] = np.where(preds == 1, "Yes", "No")
    results["Probability"] = (probs * 100).round(1).astype(str) + " %"
    results["RiskCategory"] = [f"{label_risk(p)[0]} {label_risk(p)[1]}" for p in probs]
    st.dataframe(results)

# Prediction History (Feature 6)
st.markdown("### 🕓 Prediction History")
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

if "results" in locals():
    append_df = results.copy()
else:
    append_df = input_df.copy()
    append_df["Prediction"] = "Yes" if pred else "No"
    append_df["Probability"] = f"{prob:.1%}"
    append_df["RiskCategory"] = f"{risk_emoji} {risk_label}"

st.session_state["history"] = pd.concat([st.session_state["history"], append_df], ignore_index=True)
st.dataframe(st.session_state["history"])

if st.button("🧹 Clear History"):
    st.session_state["history"] = pd.DataFrame()
