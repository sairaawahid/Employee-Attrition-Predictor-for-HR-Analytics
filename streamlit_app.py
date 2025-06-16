import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt, json
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════
# 1.  Load Cached Resources
# ═══════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("xgboost_optimized_model.pkl")

@st.cache_data
def load_schema():
    return json.loads(Path("employee_schema.json").read_text())

@st.cache_data
def load_tooltips():
    try:
        return json.loads(Path("feature_tooltips.json").read_text())
    except Exception:
        return {}

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ═══════════════════════════════════════
# 2.  Initialisation
# ═══════════════════════════════════════
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ═══════════════════════════════════════
# 3.  Helper utilities
# ═══════════════════════════════════════
def label_risk(p):
    if p < .30: return "🟢 Low"
    if p < .60: return "🟡 Moderate"
    return "🔴 High"

def safe_stats(col):
    meta = schema_meta.get(col, {})
    cmin = meta.get("min", 0.0)
    cmax = meta.get("max", 1.0)
    cmean = meta.get("mean", (cmin + cmax) / 2)
    if cmin == cmax:
        cmax = cmin + 1
    return float(cmin), float(cmax), float(cmean)

# ═══════════════════════════════════════
# 4.  UI Header
# ═══════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown("Predict attrition risk and explore explanations with **SHAP**.")

with st.expander("📘 How to use this app", expanded=False):
    st.markdown(
        """
1. **Enter employee details** in the sidebar.
2. **Use Sample Data** for a demo or **Reset Form** to start fresh.
3. **Upload a CSV (optional)** for bulk scoring and row-by-row inspection.  
4. Click **Prediction** to see risk, probability & calibrated risk category.  
5. Explore **SHAP plots** to understand which factors drive each prediction.  
6. Use the **Interactive Feature Impact** to inspect any feature.  
7. **Download or clear History** to track past predictions and share insights.
        """
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ═══════════════════════════════════════
# 5.  Sidebar widgets
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")

def sidebar_inputs():
    row = {}
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")

        if meta["dtype"] == "object":
            opts   = meta.get("options", ["Unknown"])
            curval = st.session_state.get(key, opts[0])
            if curval not in opts:
                curval = opts[0]
            row[col] = st.sidebar.selectbox(col, opts,
                                            index=opts.index(curval),
                                            key=key, help=tip)
        else:
            cmin, cmax, cmean = safe_stats(col)
            curval = float(st.session_state.get(key, cmin))
            curval = min(max(curval, cmin), cmax)   # clamp
            if abs(cmax - cmin) < 1e-9:
                row[col] = st.sidebar.number_input(col, value=curval,
                                                   key=key, help=tip)
            else:
                row[col] = st.sidebar.slider(col, cmin, cmax, curval,
                                             key=key, help=tip)
    return pd.DataFrame([row])

# ═══════════════════════════════════════
# 6.  Sample & Reset buttons
# ═══════════════════════════════════════
sample_employee = {
    "Age": 32,
    "Attrition": "No",
    "Business Travel": "Travel_Rarely",
    "Daily Rate": 1100,
    "Department": "Research & Development",
    "Distance From Home": 8,
    "Education": 3,
    "Education Field": "Life Sciences",
    "Environment Satisfaction": 3,
    "Gender": "Male",
    "Hourly Rate": 65,
    "Job Involvement": 3,
    "Job Level": 2,
    "Job Role": "Research Scientist",
    "Job Satisfaction": 2,
    "Marital Status": "Single",
    "Monthly Income": 5200,          # ← space kept
    "Monthly Rate": 14000,
    "No. of Companies Worked": 2,
    "Over Time": "Yes",
    "Percent Salary Hike": 13,
    "Performance Rating": 3,
    "Relationship Satisfaction": 2,
    "Stock Option Level": 1,
    "Total Working Years": 10,
    "Training Times Last Year": 3,
    "Work Life Balance": 2,
    "Years At Company": 5,
    "Years In Current Role": 3,
    "Years Since Last Promotion": 1,
    "Years With Current Manager": 2,
}

def load_sample():
    """Push sample values into sidebar widgets, clamping numerics where needed."""
    for col, val in sample_employee.items():
        # skip if the column isn't in the schema (prevents KeyError)
        if col not in schema_meta:
            continue

        # Clamp numeric values to slider range
        if schema_meta[col]["dtype"] != "object" and isinstance(val, (int, float)):
            cmin, cmax, _ = safe_stats(col)
            val = max(min(val, cmax), cmin)

        st.session_state[f"inp_{col}"] = val


def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        if meta["dtype"] == "object":
            st.session_state[key] = meta.get("options", ["Unknown"])[0]
        else:
            st.session_state[key] = safe_stats(col)[0]

st.sidebar.button("🧭 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form", on_click=reset_form)

# ═══════════════════════════════════════
# 7.  Gather user or batch data
# ═══════════════════════════════════════
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    batch_mode = True
else:
    raw_df = sidebar_inputs()
    batch_mode = False

# ═══════════════════════════════════════
# 8.  Prepare data for model
# ═══════════════════════════════════════
template = {c: (m["options"][0] if m["dtype"] == "object" else 0)
            for c, m in schema_meta.items()}
schema_df = pd.DataFrame([template])

X_full = pd.concat([raw_df, schema_df], ignore_index=True)
X_enc  = pd.get_dummies(X_full).iloc[:len(raw_df)]
X_enc  = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

# ═══════════════════════════════════════
# 9.  Batch prediction (if any)
# ═══════════════════════════════════════
if batch_mode:
    preds  = model.predict(X_enc)
    probs  = model.predict_proba(X_enc)[:, 1]
    raw_df["Prediction"]    = np.where(preds == 1, "Yes", "No")
    raw_df["Probability"]   = (probs*100).round(1).astype(str) + " %"
    raw_df["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("📑 Batch Prediction Summary")
    st.info("This table summarizes attrition predictions for all uploaded employees. "
        "Each row shows whether the employee is predicted to leave (Yes/No), "
        "the exact probability, and the assigned risk category: "
        "**Low (<30%)**, **Moderate (30–60%)**, or **High (>60%)**. "
        "You can select a row for detailed SHAP analysis.")
    
    sel_row = st.number_input("Row to inspect (1-based)",
                              min_value=1, max_value=len(raw_df), value=1)
    st.dataframe(raw_df)
    X_user  = X_enc.iloc[[sel_row-1]]
    user_df = raw_df.iloc[[sel_row-1]]
else:
    X_user  = X_enc.iloc[[0]]
    user_df = raw_df.iloc[[0]]

# ═══════════════════════════════════════
# 10.  Single prediction display
# ═══════════════════════════════════════
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk = label_risk(prob)

st.markdown("### 🎯 Prediction")
st.info("This section shows whether the employee is likely to leave the company (Yes/No), "
        "the exact probability of attrition, and a categorized risk level: "
        "**Low (<30%)**, **Moderate (30–60%)**, or **High (>60%)**.")

# Styled box container
st.markdown("""
<div style='border: 2px solid #eee; border-radius: 10px; padding: 20px; background-color: #f9f9f9;'>
    <div style='display: flex; justify-content: space-between; font-size: 18px;'>
        <div>
            <strong>Prediction</strong><br>
            <span style='font-size: 24px; color: #444;'>{}</span>
        </div>
        <div>
            <strong>Probability</strong><br>
            <span style='font-size: 24px; color: #444;'>{:.1%}</span>
        </div>
        <div>
            <strong>Risk Category</strong><br>
            <span style='font-size: 24px;'>{}</span>
        </div>
    </div>
</div>
""".format("Yes" if pred else "No", prob, risk),
unsafe_allow_html=True)

# ═══════════════════════════════════════
# 11.  SHAP explanations
# ═══════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
st.info("These plots show **which features push the prediction higher or lower.** "
        "Positive values increase attrition risk; negative values decrease it.")
sv = explainer.shap_values(X_user)
if isinstance(sv, list):
    sv = sv[1]


st.markdown("### 🌐 Global Impact — Beeswarm")
st.info("This plot shows which features **had the highest overall impact** "
        "on the model’s prediction for this employee. Longer bars = stronger effect. "
        "Colors indicate whether the value pushed the prediction higher (red) or lower (blue).")
fig_bsw, _ = plt.subplots()
shap.summary_plot(sv, X_user, show=False)
st.pyplot(fig_bsw); plt.clf()


st.markdown("### 🧭 Decision Path")
st.info("This plot explains the **sequence of contributions** each feature made, "
        "starting from the model’s baseline prediction. Features that increased or "
        "decreased the risk are shown from left to right, helping you follow the model’s logic.")
fig_dec, _ = plt.subplots()
shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
st.pyplot(fig_dec); plt.clf()


st.markdown("### 🎯 Local Force Plot")
st.info("This plot provides a **visual tug-of-war**: features pushing the prediction "
        "higher (red) vs. lower (blue). It gives an intuitive sense of what tipped the balance "
        "towards a high or low attrition risk for this specific case.")
try:
    fig_f = shap.plots.force(explainer.expected_value, sv[0], X_user.iloc[0],
                             matplotlib=True, show=False)
    st.pyplot(fig_f)
except Exception:
    st.info("Force plot unavailable – showing waterfall instead.")
    fig_f, _ = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=sv[0], base_values=explainer.expected_value,
                         data=X_user.iloc[0]),
        max_display=15, show=False)
    st.pyplot(fig_f)
    # st.warning("⚠️ Force plot not supported in this environment.")


st.markdown("### 🔎 Interactive Feature Impact")
st.info("Select a feature to see **how much it individually influenced** the prediction. "
        "This bar shows whether the chosen feature increased or decreased attrition risk "
        "and by how much in the context of this specific employee.")
feature = st.selectbox("Choose feature", X_user.columns, key="feat_sel")
idx     = X_user.columns.get_loc(feature)
single_val_arr = np.array([sv[0][idx]])

fig_bar, _ = plt.subplots()
shap.bar_plot(
    shap_values=single_val_arr,
    feature_names=[feature],
    max_display=1,
    show=False,
)
st.pyplot(fig_bar); plt.clf()

# ═══════════════════════════════════════
# 12.  History (append & display)
# ═══════════════════════════════════════
user_df = user_df.copy()
user_df["Prediction"]    = "Yes" if pred else "No"
user_df["Probability"]   = f"{prob:.1%}"
user_df["Risk Category"] = risk
user_df["Timestamp"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state["history"] = pd.concat(
    [st.session_state["history"], user_df], ignore_index=True
)

st.subheader("📥 Prediction History")
st.dataframe(st.session_state["history"])
csv_hist = st.session_state["history"].to_csv(index=False).encode()
st.download_button("💾 Download History", csv_hist,
                   "prediction_history.csv", "text/csv")
if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
