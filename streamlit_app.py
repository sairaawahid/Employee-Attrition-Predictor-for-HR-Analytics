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
model            = load_model()
schema           = load_schema()
X_stats          = load_stats()
feature_tooltips = load_tooltips()
explainer        = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  PAGE HEADER
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar *or* upload a CSV file.  
        2. View **attrition prediction**, probability, and color-coded risk card.  
        3. Use the **row selector** (when CSV uploaded) to inspect any employee.  
        4. Scroll to **SHAP charts** and the **Interactive Feature Impact Viewer** for deeper insights.
        """
    )

# ──────────────────────────────────────────────────────────────
# 4.  SIDEBAR INPUT FORM
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

def user_input_features() -> pd.DataFrame:
    data = {}
    for col in schema.columns:
        base = col.split("_")[0]                 # base name for tooltip lookup
        tip  = feature_tooltips.get(base, "")
        if schema[col].dtype == "object":
            data[col] = st.sidebar.selectbox(col, schema[col].unique(), help=tip)
        else:
            if col in X_stats:
                cmin, cmax, cmean = (
                    float(X_stats[col]["min"]),
                    float(X_stats[col]["max"]),
                    float(X_stats[col]["mean"]),
                )
            else:
                cmin, cmax, cmean = 0.0, 1.0, 0.5
            data[col] = (
                st.sidebar.number_input(col, value=cmin, help=tip)
                if cmin == cmax
                else st.sidebar.slider(col, cmin, cmax, cmean, help=tip)
            )
    return pd.DataFrame(data, index=[0])

# ──────────────────────────────────────────────────────────────
# 5.  CSV UPLOAD OR SINGLE ENTRY
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload Employee CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.success(f"Loaded **{len(raw_df)}** employees from CSV.")
else:
    raw_df = user_input_features()

# One-hot encode & align with training columns
X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_pred = X_enc.iloc[: len(raw_df)]

# ──────────────────────────────────────────────────────────────
# 6.  PREDICTIONS & RISK CARDS / TABLE
# ──────────────────────────────────────────────────────────────
preds = model.predict(X_pred)
probs = model.predict_proba(X_pred)[:, 1]

if len(raw_df) == 1:
    pred, prob = preds[0], probs[0]
    risk_label = "🟢 Low" if prob < 0.30 else "🟡 Moderate" if prob < 0.60 else "🔴 High"

    st.subheader("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "Yes" if pred else "No")
    c2.metric("Attrition Probability", f"{prob:.1%}")
    c3.metric("Risk Category", risk_label)

else:  # batch mode
    results = raw_df.copy()
    results["Prediction"]  = np.where(preds == 1, "Yes", "No")
    results["Probability"] = (probs * 100).round(1).astype(str) + " %"
    results["Risk"]        = pd.cut(probs, [0, .3, .6, 1], labels=["Low", "Moderate", "High"])
    st.subheader("📑 Batch Predictions")
    st.dataframe(results)

# ──────────────────────────────────────────────────────────────
# 7.  ROW SELECTOR FOR SHAP INSPECTION
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

row_idx = 0
if len(raw_df) > 1:
    row_idx = st.number_input("Select employee row to inspect:",
                              min_value=0, max_value=len(raw_df) - 1, step=1, value=0)
X_row       = X_pred.iloc[[row_idx]]
shap_values = explainer.shap_values(X_row)
if not isinstance(shap_values, np.ndarray):  # list for binary
    shap_values = shap_values[1]

# ───── 7-A Beeswarm
st.markdown("### 🌐 Global Impact — Beeswarm")
fig_bee, _ = plt.subplots()
shap.summary_plot(shap_values, X_row, show=False)
st.pyplot(fig_bee)
plt.clf()

# ───── 7-B Decision plot
st.markdown("### 🧭 Decision Path (Individual)")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_values[0], X_row, show=False)
st.pyplot(fig_dec)
plt.clf()

# ───── 7-C Force plot
st.markdown("### 🎯 Local Force Plot")
fig_force = shap.plots.force(
    explainer.expected_value, shap_values[0], X_row.iloc[0],
    matplotlib=True, show=False
)
st.pyplot(fig_force)

# ──────────────────────────────────────────────────────────────
# 8.  FEATURE 5 – INTERACTIVE FEATURE IMPACT VIEWER
# ──────────────────────────────────────────────────────────────
st.markdown("## 🔬 Interactive Feature Impact Viewer")

# Rank features by |SHAP|
abs_shap = np.abs(shap_values[0])
sorted_idx = np.argsort(abs_shap)[::-1]
sorted_features = [X_row.columns[i] for i in sorted_idx]

selected_feature = st.selectbox(
    "Choose a feature to view its SHAP contribution:",
    sorted_features
)

feat_idx = list(X_row.columns).index(selected_feature)
feat_value = X_row.iloc[0, feat_idx]
feat_shap  = shap_values[0][feat_idx]

# Simple bar plot: single feature contribution
fig_feat, ax_feat = plt.subplots(figsize=(4, 1.2))
color = "red" if feat_shap > 0 else "blue"
ax_feat.barh([selected_feature], [feat_shap], color=color)
ax_feat.set_xlabel("SHAP value (impact on log-odds)")
ax_feat.set_xlim(min(0, feat_shap) * 1.2, max(0, feat_shap) * 1.2)
ax_feat.axvline(0, color="k", linewidth=.8)
ax_feat.set_yticklabels([f"{selected_feature} = {feat_value}"])
st.pyplot(fig_feat)
plt.clf()

st.caption("▲ Positive SHAP pushes toward leaving; ▼ Negative pushes toward staying.")

# ──────────────────────────────────────────────────────────────
# 9.  PSYCHOLOGY-BASED HR RECOMMENDATIONS
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
