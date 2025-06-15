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
tooltips = load_tooltips()
explainer = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  TITLE & GUIDE
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar or **upload a CSV** for batch predictions.  
        2. Results update with **risk prediction & explanations**.  
        3. Use SHAP charts to understand feature importance.  
        4. Inspect individual employees from the uploaded CSV.  
        5. Use the psychology-based suggestions to guide HR strategy.
        """
    )

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 4.  INPUT OR CSV UPLOAD
# ──────────────────────────────────────────────────────────────
def user_input_features():
    data = {}
    for col in schema.columns:
        base = col.split("_")[0]
        hint = tooltips.get(base, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique(), help=hint)
        else:
            if col in X_stats:
                cmin, cmax, cmean = map(float, [X_stats[col]["min"], X_stats[col]["max"], X_stats[col]["mean"]])
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = st.sidebar.slider(col, cmin, cmax, cmean, help=hint)
    return pd.DataFrame(data, index=[0])

uploaded = st.sidebar.file_uploader("📂 Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df_full = pd.concat([df, schema]).drop_duplicates(keep="first")
    X_all = pd.get_dummies(df_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_all.iloc[:len(df)]
    preds = model.predict(X_user)
    probs = model.predict_proba(X_user)[:, 1]
    df["Prediction"] = np.where(preds == 1, "Yes", "No")
    df["Attrition Probability"] = (probs * 100).round(1).astype(str) + "%"
    st.subheader("📊 Batch Predictions")
    index = st.selectbox("Select employee row:", df.index)
    input_df = df.iloc[[index]].drop(columns=["Prediction", "Attrition Probability"], errors="ignore")
    X_selected = X_user.iloc[[index]]
    pred, prob = preds[index], probs[index]
    st.dataframe(df)
else:
    input_df = user_input_features()
    X_base = pd.concat([input_df, schema]).drop_duplicates(keep="first")
    X_all = pd.get_dummies(X_base).reindex(columns=schema.columns, fill_value=0)
    X_selected = X_all.iloc[[0]]
    pred = model.predict(X_selected)[0]
    prob = model.predict_proba(X_selected)[0, 1]

# ──────────────────────────────────────────────────────────────
# 5.  METRIC CARDS
# ──────────────────────────────────────────────────────────────
st.subheader("Prediction")

risk = "🟢 Low" if prob < 0.3 else "🟡 Moderate" if prob < 0.6 else "🔴 High"
col1, col2, col3 = st.columns(3)
col1.metric("Prediction", "Yes" if pred else "No")
col2.metric("Probability", f"{prob:.1%}")
col3.metric("Risk Category", risk)

# ──────────────────────────────────────────────────────────────
# 6.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

shap_vals = explainer.shap_values(X_selected)
shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, ax1 = plt.subplots()
shap.summary_plot(shap_vals, X_selected, show=False)
st.pyplot(fig1)
plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
fig2, ax2 = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_selected, show=False)
st.pyplot(fig2)
plt.clf()

st.markdown("### 🎯 Local Force Plot")
fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_selected.iloc[0], matplotlib=True, show=False)
st.pyplot(fig3)
st.caption("▲ Positive SHAP pushes toward leaving; ▼ Negative pushes toward staying.")

# ──────────────────────────────────────────────────────────────
# 7.  PSYCHOLOGY-BASED HR RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🧠 Psychology-Based HR Recommendations")

tips = []
rec = {
    "JobSatisfaction": {
        1: "Very low job satisfaction. Consider role changes or engagement programs.",
        2: "Moderate dissatisfaction — explore internal mobility or mentoring.",
        3: "Generally satisfied. Sustain engagement.",
        4: "Highly satisfied — continue current support."
    },
    "EnvironmentSatisfaction": {
        1: "Poor environment rating. Check ergonomics or team climate.",
        2: "Mediocre satisfaction — ask for feedback.",
        3: "Supportive environment likely.",
        4: "Excellent satisfaction with work setting."
    },
    "RelationshipSatisfaction": {
        1: "Poor peer relations — offer team-building or communication training.",
        2: "May benefit from interpersonal coaching.",
        3: "Workplace climate appears fair.",
        4: "Strong relationships — a retention strength."
    },
    "JobInvolvement": {
        1: "Low involvement. Try goal setting or recognition.",
        2: "Could improve with motivation or autonomy.",
        3: "Engaged employee.",
        4: "Highly involved — sustain momentum."
    },
    "WorkLifeBalance": {
        1: "Work-life conflict. Consider flexible work policies.",
        2: "At risk of imbalance — check workloads.",
        3: "Balance is healthy.",
        4: "Excellent balance — reinforce this culture."
    },
    "OverTime_Yes": "Regular overtime flagged. Watch for burnout or overload."
}

for f in ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement", "WorkLifeBalance"]:
    score = input_df.get(f, [None])[0]
    if score in rec[f]:
        tips.append(rec[f][score])

if "OverTime_Yes" in X_selected.columns and X_selected["OverTime_Yes"].iloc[0] == 1:
    tips.append(rec["OverTime_Yes"])

if tips:
    for msg in tips:
        st.info(msg)
else:
    st.success("No major psychological flags. Maintain support and communication.")
