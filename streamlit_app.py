import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re
from datetime import datetime

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
model          = load_model()
schema         = load_schema()
X_stats        = load_stats()
tooltips       = load_tooltips()
explainer      = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()
if "chat" not in st.session_state:
    st.session_state["chat"] = []                # store (is_user, text)

# ──────────────────────────────────────────────────────────────
# 3.  RISK CATEGORY FUNCTION
# ──────────────────────────────────────────────────────────────
def label_risk(p: float) -> str:
    if p < 0.30:
        return "🟢 Low"
    elif p < 0.60:
        return "🟡 Moderate"
    else:
        return "🔴 High"

# ──────────────────────────────────────────────────────────────
# 4.  PAGE HEADER / GUIDE / UPLOAD
# ──────────────────────────────────────────────────────────────
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore model explanations with **SHAP**.")

with st.expander("📘 How to use this app"):
    st.markdown(
        """
        1. **Enter employee details** in the sidebar or **upload a CSV** for batch scoring.  
        2. The panel shows **attrition prediction, probability, risk category** and SHAP insights.  
        3. Use **Interactive Feature Impact** (dropdown) to inspect any feature.  
        4. Download or clear **prediction history** at any time.  
        5. Ask the **HR Chatbot** questions like “How can we retain this employee?”.
        """
    )
    uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

st.sidebar.header("📋 Employee Attributes")

# ──────────────────────────────────────────────────────────────
# 5.  SIDEBAR INPUT WITH TOOLTIPS
# ──────────────────────────────────────────────────────────────
def user_input_features() -> pd.DataFrame:
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

# ──────────────────────────────────────────────────────────────
# 6.  DETERMINE INPUT MODE
# ──────────────────────────────────────────────────────────────
batch_mode = False
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
    X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_enc.iloc[[0]]
    batch_mode = True
else:
    raw_df = user_input_features()
    X_full = pd.concat([raw_df, schema]).drop_duplicates(keep="first")
    X_enc  = pd.get_dummies(X_full).reindex(columns=schema.columns, fill_value=0)
    X_user = X_enc.iloc[[0]]

# ──────────────────────────────────────────────────────────────
# 7.  PREDICT & RISK LABEL
# ──────────────────────────────────────────────────────────────
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk_cat = label_risk(prob)

# ──────────────────────────────────────────────────────────────
# 8.  METRIC CARDS
# ──────────────────────────────────────────────────────────────
st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "Yes" if pred else "No")
c2.metric("Probability", f"{prob:.1%}")
c3.metric("Risk Category", risk_cat)

# ──────────────────────────────────────────────────────────────
# 9.  SHAP EXPLANATIONS
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanations")

shap_vals = explainer.shap_values(X_user)
if isinstance(shap_vals, list):  # tree explainer returns list [neg, pos]
    shap_vals = shap_vals[1]

st.markdown("### 🌐 Global Impact — Beeswarm")
fig1, _ = plt.subplots()
shap.summary_plot(shap_vals, X_user, show=False)
st.pyplot(fig1); plt.clf()

st.markdown("### 🧭 Decision Path (Individual)")
fig2, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, shap_vals[0], X_user, show=False)
st.pyplot(fig2); plt.clf()

st.markdown("### 🎯 Local Force Plot")
try:
    fig3 = shap.plots.force(explainer.expected_value, shap_vals[0], X_user.iloc[0],
                            matplotlib=True, show=False)
    st.pyplot(fig3)
except Exception:
    st.write("Force plot could not render; showing waterfall instead.")
    fig3, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=shap_vals[0],
                         base_values=explainer.expected_value,
                         data=X_user.iloc[0]),
        max_display=15, show=False)
    st.pyplot(fig3)
st.caption("Positive SHAP values push toward leaving; negative values push toward staying.")

# ──────────────────────────────────────────────────────────────
# 10.  OPTIONAL BATCH VIEW
# ──────────────────────────────────────────────────────────────
if batch_mode:
    preds = model.predict(X_enc)
    probs = model.predict_proba(X_enc)[:, 1]
    out = raw_df.copy()
    out["Prediction"]       = np.where(preds == 1, "Yes", "No")
    out["Probability"]      = (probs * 100).round(1).astype(str) + " %"
    out["Risk Category"]    = [label_risk(p) for p in probs]
    st.markdown("### 📑 Batch Prediction Summary")
    st.dataframe(out)

# ──────────────────────────────────────────────────────────────
# 11.  INTERACTIVE FEATURE IMPACT VIEWER
# ──────────────────────────────────────────────────────────────
st.markdown("## 🔬 Interactive Feature Impact Viewer")
abs_vals   = np.abs(shap_vals[0])
sorted_idx = abs_vals.argsort()[::-1]
feature_list = [X_user.columns[i] for i in sorted_idx]
sel_feat = st.selectbox("Select a feature to inspect:", feature_list)

i = list(X_user.columns).index(sel_feat)
val = X_user.iloc[0, i]
sv  = shap_vals[0][i]

fig_feat, ax = plt.subplots(figsize=(4, 1.2))
ax.barh([sel_feat], [sv], color="red" if sv > 0 else "blue")
ax.axvline(0, color="k", lw=.8)
ax.set_yticklabels([f"{sel_feat} = {val}"])
ax.set_xlabel("SHAP value (impact on log-odds)")
st.pyplot(fig_feat); plt.clf()

# ──────────────────────────────────────────────────────────────
# 12.  PSYCHOLOGY-BASED RECOMMENDATIONS (unchanged)
# ──────────────────────────────────────────────────────────────
st.subheader("🧠 Psychology-Based HR Recommendations")
rec_map = {
    "JobSatisfaction": {
        1: "Very low job satisfaction – explore role fit or engagement.",
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
    "WorkLifeBalance": {
        1: "Work-life conflict – consider flexible policies.",
        2: "Potential imbalance – monitor workload.",
        3: "Healthy work-life balance.",
        4: "Excellent balance – keep supporting."
    },
    "OverTime_Yes": "Regular overtime detected – assess workload & burnout risk."
}

tips = []
def safe_val(col): return raw_df[col].iloc[0] if col in raw_df.columns else None
for col in ["JobSatisfaction","EnvironmentSatisfaction",
            "RelationshipSatisfaction","WorkLifeBalance"]:
    v = safe_val(col)
    if v in rec_map[col]: tips.append(rec_map[col][v])
if "OverTime_Yes" in X_user.columns and X_user["OverTime_Yes"].iloc[0] == 1:
    tips.append(rec_map["OverTime_Yes"])

for t in tips: st.info(t)
if not tips: st.success("No critical psychological flags detected.")

# ──────────────────────────────────────────────────────────────
# 13.  PREDICTION HISTORY (unchanged logic, now storing risk)
# ──────────────────────────────────────────────────────────────
append_df = raw_df.copy().iloc[[0]]
append_df["Prediction"]  = "Yes" if pred else "No"
append_df["Probability"] = f"{prob:.1%}"
append_df["Risk Category"] = risk_cat
append_df["Timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat([st.session_state["history"], append_df], ignore_index=True)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
hist_csv = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", hist_csv, "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()

# ──────────────────────────────────────────────────────────────
# 14.  FEATURE 11 – CHATBOT-STYLE HR GUIDANCE
# ──────────────────────────────────────────────────────────────
st.markdown("## 💬 HR Assistant Chatbot")
def assistant_reply(msg: str, risk_label: str) -> str:
    msg_lower = msg.lower()
    if "retain" in msg_lower or "keep" in msg_lower:
        base = "Focus on intrinsic motivators, career growth, and work-life balance."
        add  = ""
        if "high" in risk_label.lower():   add = "  Immediate action is advised because risk is high."
        elif "moderate" in risk_label.lower(): add = "  Consider pulse-surveys and manager check-ins."
        else: add = "  Keep up the good practices that support engagement."
        return base + add
    if "reason" in msg_lower or "why" in msg_lower:
        return "Check the top SHAP features. High positive SHAP values push the employee toward leaving."
    if "tip" in msg_lower or "suggest" in msg_lower:
        return "Offer flexible schedules, recognize achievements, and provide clear career paths."
    return "I'm here to help interpret the results and suggest retention actions. Ask me anything!"

# display chat history
for is_user, txt in st.session_state["chat"]:
    align = "user" if is_user else "assistant"
    with st.chat_message(align):
        st.markdown(txt)

prompt = st.chat_input("Ask the HR Assistant…")
if prompt:
    st.session_state["chat"].append((True, prompt))
    response = assistant_reply(prompt, risk_cat)
    st.session_state["chat"].append((False, response))
    st.rerun()
