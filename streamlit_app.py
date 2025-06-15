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

@st.cache_data
def load_tooltips():
    return json.loads(Path("feature_tooltips.json").read_text())

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ──────────────────────────────────────────────────────────────
# 2.  INITIALIZE
# ──────────────────────────────────────────────────────────────
model   = load_model()
schema  = load_schema()
X_stats = load_stats()
feature_tooltips = load_tooltips()
explainer = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 3.  RISK CATEGORY LABEL FUNCTION
# ──────────────────────────────────────────────────────────────
def label_risk(prob):
    if prob < 0.30:
        return "🟢 Low"
    elif prob < 0.60:
        return "🟡 Moderate"
    else:
        return "🔴 High"

# ──────────────────────────────────────────────────────────────
# 4.  TITLE / GUIDE / UPLOAD SECTION
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        '''
        1. **Enter employee details** in the sidebar or **upload a CSV** to score multiple employees.  
        2. The main panel updates with **attrition risk & probability**.  
        3. Scroll to **SHAP charts** to see which features drive the prediction.  
        4. Use the row selector (when a CSV is uploaded) to inspect individual employees.  
        5. Use these insights to design targeted HR interventions.
        '''
    )
    uploaded_file = st.file_uploader("📤 Upload Your Own CSV (Batch Prediction)", type="csv")

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 5.  INPUT FUNCTION WITH TOOLTIP SUPPORT
# ──────────────────────────────────────────────────────────────
def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        base_col = col.split("_")[0]
        tooltip = feature_tooltips.get(base_col, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique(), help=tooltip)
        else:
            if col in X_stats:
                cmin = float(X_stats[col]["min"])
                cmax = float(X_stats[col]["max"])
                cmean = float(X_stats[col]["mean"])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = st.sidebar.slider(col, cmin, cmax, cmean, help=tooltip)
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 6.  MAIN LOGIC – SINGLE PREDICTION OR CSV BATCH
# ──────────────────────────────────────────────────────────────
use_batch = False
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    X_full = pd.concat([uploaded_df, schema]).drop_duplicates(keep="first")
    X_encoded = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_encoded.iloc[0:1]
    use_batch = True
else:
    input_df = user_input_features()
    X_full = pd.concat([input_df, schema]).drop_duplicates(keep="first")
    X_encoded = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_encoded.iloc[[0]]

# ──────────────────────────────────────────────────────────────
# 7.  PREDICTION + RISK LABELING
# ──────────────────────────────────────────────────────────────
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk_category = label_risk(prob)

# ──────────────────────────────────────────────────────────────
# 8.  METRIC CARDS
# ──────────────────────────────────────────────────────────────
st.subheader("Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Prediction", "Yes" if pred else "No")
col2.metric("Attrition Probability", f"{prob:.1%}")
col3.metric("Risk Category", risk_category)

# ──────────────────────────────────────────────────────────────
# 9.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")
shap_vals = explainer.shap_values(X_user)
shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, ax1 = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig1)
plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
fig2, ax2 = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(fig2)
plt.clf()

st.markdown("### 🎯 Local Force Plot")
fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0], matplotlib=True, show=False)
st.pyplot(fig3)
st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")

# ──────────────────────────────────────────────────────────────
# 10.  OPTIONAL: CSV BATCH MODE VIEW / INSPECTION
# ──────────────────────────────────────────────────────────────
if use_batch:
    results = uploaded_df.copy()
    results["Prediction"] = model.predict(X_encoded)
    results["Attrition Probability"] = model.predict_proba(X_encoded)[:, 1]
    results["Prediction"] = results["Prediction"].map({1: "Yes", 0: "No"})
    results["Risk Category"] = model.predict_proba(X_encoded)[:, 1].apply(label_risk)
    results["Attrition Probability"] = results["Attrition Probability"].apply(lambda p: f"{p:.1%}")

    st.markdown("### 📊 Batch Prediction Summary")
    st.dataframe(results)

    selected_row = st.selectbox("Select employee for SHAP inspection", results.index)
    if selected_row is not None:
        X_user = X_encoded.iloc[[selected_row]]
        shap_vals = explainer.shap_values(X_user)
        shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]

        st.markdown("### 🔍 SHAP Re-Inspection")
        fig4, ax4 = plt.subplots()
        shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
        st.pyplot(fig4)
        plt.clf()

# ──────────────────────────────────────────────────────────────
# 11.  DOWNLOADABLE HISTORY
# ──────────────────────────────────────────────────────────────
append_df = input_df.copy() if not use_batch else uploaded_df.iloc[[selected_row]]
append_df["Prediction"] = "Yes" if pred else "No"
append_df["Attrition Probability"] = f"{prob:.1%}"
append_df["Risk Category"] = risk_category
st.session_state["history"] = pd.concat([st.session_state["history"], append_df], ignore_index=True)

st.markdown("### 📥 Download Prediction History")
if not st.session_state["history"].empty:
    st.dataframe(st.session_state["history"])
    csv_data = st.session_state["history"].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_data, "prediction_history.csv", "text/csv")

    if st.button("Clear History"):
        st.session_state["history"] = pd.DataFrame()
