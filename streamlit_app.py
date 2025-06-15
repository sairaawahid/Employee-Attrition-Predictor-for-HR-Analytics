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
model      = load_model()
schema     = load_schema()
X_stats    = load_stats()
tooltips   = load_tooltips()
explainer  = get_explainer(model)

# ──────────────────────────────────────────────────────────────
# 3.  HEADER & GUIDE
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar or **upload a CSV** below.  
        2. View predicted **attrition risk, probability & risk card**.  
        3. Use the row selector (for CSV) to inspect any employee.  
        4. Explore **SHAP charts** and interactively inspect single-feature impact.  
        5. Download the full prediction history at the bottom.
        """
    )

# ──────────────────────────────────────────────────────────────
# 4.  CSV UPLOAD (main panel)
# ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload Employee CSV (optional)", type="csv")

# ──────────────────────────────────────────────────────────────
# 5.  SIDEBAR INPUT FORM
# ──────────────────────────────────────────────────────────────
st.sidebar.header("📋 Employee Attributes")

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

# Build input frame
if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.success(f"Loaded **{len(raw_df)}** employees from CSV.")
else:
    raw_df = user_input_features()

# One-hot encode
X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
X_pred = X_enc.iloc[: len(raw_df)]

# ──────────────────────────────────────────────────────────────
# 6.  PREDICT
# ──────────────────────────────────────────────────────────────
preds = model.predict(X_pred)
probs = model.predict_proba(X_pred)[:, 1]

if len(raw_df) > 1:  # batch
    results = raw_df.copy()
    results["Prediction"]  = np.where(preds == 1, "Yes", "No")
    results["Probability"] = (probs * 100).round(1).astype(str) + " %"
    results["Risk"]        = pd.cut(probs, [0, .3, .6, 1], labels=["Low", "Moderate", "High"])
    st.subheader("📑 Batch Predictions")
    st.dataframe(results)

    row_idx = st.number_input("Select employee row to inspect:", 0, len(raw_df)-1, 0)
    input_df = raw_df.iloc[[row_idx]]
    X_user   = X_pred.iloc[[row_idx]]
    pred, prob = preds[row_idx], probs[row_idx]
else:               # single
    input_df = raw_df
    X_user   = X_pred
    pred, prob = preds[0], probs[0]

# ──────────────────────────────────────────────────────────────
# 7.  METRIC CARDS
# ──────────────────────────────────────────────────────────────
st.subheader("Prediction")
risk_tag = "🟢 Low" if prob < .3 else "🟡 Moderate" if prob < .6 else "🔴 High"
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk_tag)

# ──────────────────────────────────────────────────────────────
# 8.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

shap_vals = explainer.shap_values(X_user)
shap_vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig1); plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
fig2, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(fig2); plt.clf()

st.markdown("### 🎯 Local Force Plot")
fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0],
                        matplotlib=True, show=False)
st.pyplot(fig3)

# ──────────────────────────────────────────────────────────────
# 9.  🔬 INTERACTIVE FEATURE IMPACT VIEWER  (NEW)
# ──────────────────────────────────────────────────────────────
st.markdown("## 🔬 Interactive Feature Impact Viewer")

# Rank features by absolute SHAP value
abs_vals = np.abs(shap_vals[0])
sorted_idx = abs_vals.argsort()[::-1]
feature_list = [X_user.columns[i] for i in sorted_idx]

sel_feature = st.selectbox("Select a feature to inspect:", feature_list)

# bar showing that feature's SHAP value
idx = list(X_user.columns).index(sel_feature)
feat_val  = X_user.iloc[0, idx]
feat_shap = shap_vals[0][idx]

fig_feat, ax_feat = plt.subplots(figsize=(4, 1.2))
color = "red" if feat_shap > 0 else "blue"
ax_feat.barh([sel_feature], [feat_shap], color=color)
ax_feat.axvline(0, color="k", linewidth=.7)
ax_feat.set_xlim(min(feat_shap, 0)*1.2, max(feat_shap, 0)*1.2)
ax_feat.set_yticklabels([f"{sel_feature} = {feat_val}"])
ax_feat.set_xlabel("SHAP value (impact on log-odds)")
st.pyplot(fig_feat); plt.clf()

# ──────────────────────────────────────────────────────────────
# 10.  PSYCHOLOGY-BASED HR RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🧠 Psychology-Based HR Recommendations")

rec_map = {
    "JobSatisfaction": {
        1: "Very low job satisfaction – explore role fit or engagement programs.",
        2: "Moderate dissatisfaction – mentoring or job enrichment may help.",
        3: "Generally satisfied – maintain engagement.",
        4: "Highly satisfied – continue supporting growth."
    },
    "EnvironmentSatisfaction": {
        1: "Poor environment rating – review ergonomics and team climate.",
        2: "Mediocre rating – gather feedback for improvements.",
        3: "Supportive environment.",
        4: "Excellent environment satisfaction."
    },
    "RelationshipSatisfaction": {
        1: "Poor coworker relations – consider team-building or mediation.",
        2: "Average relations – encourage open communication.",
        3: "Healthy coworker relations.",
        4: "Strong relationships – leverage for mentoring."
    },
    "JobInvolvement": {
        1: "Low involvement – clarify goals and recognize achievements.",
        2: "Could benefit from intrinsic motivators.",
        3: "Good engagement.",
        4: "Highly involved – potential leader."
    },
    "WorkLifeBalance": {
        1: "Work-life conflict – consider flexibility and workload review.",
        2: "At risk of imbalance – monitor hours.",
        3: "Healthy balance.",
        4: "Excellent work-life balance."
    },
    "OverTime_Yes": "Regular overtime detected – assess workload and burnout risk."
}

def safe_val(col):
    return input_df[col].iloc[0] if col in input_df.columns else None

tips = []
for col in ["JobSatisfaction","EnvironmentSatisfaction",
            "RelationshipSatisfaction","JobInvolvement","WorkLifeBalance"]:
    v = safe_val(col)
    if v in rec_map[col]: tips.append(rec_map[col][v])

if "OverTime_Yes" in X_user.columns and X_user["OverTime_Yes"].iloc[0] == 1:
    tips.append(rec_map["OverTime_Yes"])

for txt in tips: st.info(txt)
if not tips: st.success("No critical psychological flags found.")

# ──────────────────────────────────────────────────────────────
# 11.  PREDICTION HISTORY + DOWNLOAD
# ──────────────────────────────────────────────────────────────
if "pred_history" not in st.session_state:
    st.session_state["pred_history"] = pd.DataFrame()

if len(raw_df) > 1:
    append_df = results.copy()
else:
    append_df = input_df.copy()
    append_df["Prediction"]  = "Yes" if pred else "No"
    append_df["Probability"] = f"{prob:.1%}"
    append_df["Risk"]        = risk_tag.split()[1]

st.session_state["pred_history"] = pd.concat(
    [st.session_state["pred_history"], append_df], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["pred_history"])

csv = st.session_state["pred_history"].to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Prediction History CSV", csv,
                   file_name="prediction_history.csv", mime="text/csv")

if st.button("🗑️ Clear History"):
    st.session_state["pred_history"] = pd.DataFrame()
    st.experimental_rerun()
