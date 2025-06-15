import streamlit as st
import pandas as pd
import numpy as np
import shap
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="Employee Attrition Predictor")

# Load model and metadata
model = joblib.load("model.pkl")
with open("employee_schema.json") as f:
    schema = json.load(f)
with open("feature_tooltips.json") as f:
    tooltips = json.load(f)

# Utility: infer dtype from schema
def is_categorical(col):
    return isinstance(schema[col].get("dtype", ""), str) and schema[col]["dtype"] == "object"

def sidebar_inputs():
    st.markdown("### 🎛️ Enter Employee Attributes")
    data = {}
    for col, meta in schema.items():
        key = f"input_{col}"
        tip = tooltips.get(col, "")

        if is_categorical(col):
            options = meta.get("options", [])
            default = st.session_state.get(key, options[0] if options else "")
            data[col] = st.selectbox(col, options, index=options.index(default) if default in options else 0, key=key, help=tip)
        else:
            cmin, cmax, cmean = meta["min"], meta["max"], meta["mean"]
            is_int = str(meta["dtype"]).startswith("int")
            cast = (lambda x: int(round(x))) if is_int else float
            cmin, cmax, cmean = cast(cmin), cast(cmax), cast(cmean)
            session_val = st.session_state.get(key, cmean)
            try:
                session_val = cast(session_val)
            except Exception:
                session_val = cmean
            session_val = min(max(session_val, cmin), cmax)

            if cmin == cmax:
                data[col] = st.number_input(col, value=session_val, key=key, help=tip)
            else:
                try:
                    data[col] = st.slider(col, cmin, cmax, session_val, key=key, help=tip)
                except Exception as e:
                    st.warning(f"{col}: slider unsupported ({e}). Showing number box.")
                    data[col] = st.number_input(col, min_value=cmin, max_value=cmax, value=session_val, key=key, help=tip)
    return pd.DataFrame([data])

# Callback: Reset Form
def reset_form():
    for col in schema:
        st.session_state[f"input_{col}"] = schema[col].get("mean") if not is_categorical(col) else schema[col]["options"][0]

# Callback: Use Sample Data
def load_sample():
    for col in schema:
        key = f"input_{col}"
        val = schema[col]["sample"]
        st.session_state[key] = val

# UI Header
st.title("📊 Employee Attrition Predictor for HR Analytics")

with st.expander("ℹ️ How to use this app", expanded=True):
    st.markdown("""
This AI-powered HR dashboard predicts the likelihood of employee attrition using behavioral and organizational data. You can:

- Manually input employee attributes below
- Or upload a CSV file for batch prediction

Each prediction includes SHAP visual explanations and psychology-based HR recommendations.
""")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔁 Reset Form", on_click=reset_form)
    with col2:
        st.button("🧪 Use Sample Data", on_click=load_sample)

# Upload CSV or single mode
uploaded_file = st.file_uploader("📤 Upload CSV for Batch Prediction (optional)", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# Prediction
if st.button("🔮 Predict Attrition"):
    df = pd.get_dummies(raw_df)
    for col in model.feature_names_in_:
        if col not in df:
            df[col] = 0
    df = df[model.feature_names_in_]

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    result_df = raw_df.copy()
    result_df["Attrition Prediction"] = np.where(preds == 1, "Yes", "No")
    result_df["Attrition Probability (%)"] = (probs * 100).round(2)
    result_df["Risk Category"] = pd.cut(probs, bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low Risk", "Moderate Risk", "High Risk"])

    st.subheader("🧾 Prediction Results")
    st.dataframe(result_df)

    # SHAP
    st.subheader("🧠 SHAP Explanations")
    explainer = shap.Explainer(model)
    shap_vals = explainer(df)
    st.pyplot(shap.plots.beeswarm(shap_vals, show=False))

    if not batch_mode:
        st.subheader("🔍 Local Force Plot")
        fig = shap.plots.force(explainer.expected_value[1], shap_vals[0].values, df.iloc[0], matplotlib=True, show=False)
        st.pyplot(fig)

    # Download
    st.download_button("💾 Download Prediction CSV", data=result_df.to_csv(index=False), file_name="attrition_predictions.csv", mime="text/csv")

