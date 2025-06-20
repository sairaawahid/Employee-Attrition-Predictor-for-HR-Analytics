import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt, json
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════
# 1.  Cached resources
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


@st.cache_resource(hash_funcs={joblib.Bunch: lambda _: None})
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ═══════════════════════════════════════
# 2.  Initialise
# ═══════════════════════════════════════
model        = load_model()
schema_meta  = load_schema()
tooltips     = load_tooltips()
explainer    = get_explainer(model)

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# ═══════════════════════════════════════
# 3.  Helpers
# ═══════════════════════════════════════
def label_risk(p: float) -> str:
    if p < .30:
        return "🟢 Low"
    if p < .60:
        return "🟡 Moderate"
    return "🔴 High"


def safe_stats(col):
    """Return min, max, mean from schema – widen range if collapsed."""
    meta = schema_meta.get(col, {})
    cmin = meta.get("min", 0.0)
    cmax = meta.get("max", 1.0)
    cmean = meta.get("mean", (cmin + cmax) / 2)
    if cmin == cmax:
        cmax = cmin + 1
    return float(cmin), float(cmax), float(cmean)


# ═══════════════════════════════════════
# 4.  UI header & guide
# ═══════════════════════════════════════
st.title("Employee Attrition Predictor")
st.markdown(
    "A decision-support tool for HR professionals to **predict employee attrition**, "
    "run bulk analyses, and **explain every prediction with SHAP**."
)

with st.expander("**How to use this app**", expanded=False):
    st.markdown(
        """
1. Enter attributes in the **sidebar** or click **Use Sample Data**.  
2. *Optional*: upload a **CSV** for bulk scoring.  
3. Press **Run Prediction**.  
4. Review the probability, calibrated risk level and SHAP explanations.  
5. Use **Interactive Feature Impact** to inspect any feature’s contribution.  
6. Download or clear the **Prediction History** at any time.
        """
    )

uploaded_file = st.file_uploader("📂 Upload CSV (optional)", type="csv")

# ═══════════════════════════════════════
# 5.  Sidebar inputs
# ═══════════════════════════════════════
st.sidebar.header("📋 Employee Attributes")


def sidebar_inputs() -> pd.DataFrame:
    """Render widgets; return single-row DataFrame."""
    row = {}
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        tip = tooltips.get(col.split("_")[0], "")

        if meta["dtype"] == "object":
            options = meta.get("options", ["Unknown"])
            cur = st.session_state.get(key, options[0])
            if cur not in options:
                cur = options[0]
            row[col] = st.sidebar.selectbox(col, options,
                                            index=options.index(cur),
                                            key=key, help=tip)
        else:
            cmin, cmax, _ = safe_stats(col)
            cur = float(st.session_state.get(key, cmin))

            # clamp to bounds
            cur = min(max(cur, cmin), cmax)

            discrete = meta.get("discrete", False)
            if discrete:
                # cast everything to float so Streamlit types match
                cmin, cmax, cur, step = map(float, (int(cmin), int(cmax), int(cur), 1))
            else:
                step = 0.1

            if abs(cmax - cmin) < 1e-9:
                row[col] = st.sidebar.number_input(col, value=cur, key=key, help=tip)
            else:
                row[col] = st.sidebar.slider(col, cmin, cmax, cur,
                                             step=step, key=key, help=tip)
    return pd.DataFrame([row])


# ═══════════════════════════════════════
# 6.  Sample & reset buttons
# ═══════════════════════════════════════
sample_employee = {
    "Age": 32,
    "Attrition": "No",
    "Business Travel": "Travel_Rarely",
    "Daily Rate": 1100,
    "Department": "Research & Development",
    "Distance From Home": 8,
    "Education": "Bachelor's",
    "Education Field": "Life Sciences",
    "Environment Satisfaction": 3,
    "Gender": "Male",
    "Hourly Rate": 65,
    "Job Involvement": 3,
    "Job Level": 2,
    "Job Role": "Research Scientist",
    "Job Satisfaction": 2,
    "Marital Status": "Single",
    "Monthly Income": "5 000 – 5 999",
    "Monthly Rate": "10 000 – 14 999",
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
    for col, val in sample_employee.items():
        if col not in schema_meta:
            continue
        if schema_meta[col]["dtype"] != "object":
            cmin, cmax, _ = safe_stats(col)
            val = max(min(val, cmax), cmin)
        st.session_state[f"inp_{col}"] = val


def reset_form():
    for col, meta in schema_meta.items():
        key = f"inp_{col}"
        if meta["dtype"] == "object":
            st.session_state[key] = meta.get("options", ["Unknown"])[0]
        else:
            st.session_state[key] = safe_stats(col)[0]   # min


st.sidebar.button("📄 Use Sample Data", on_click=load_sample)
st.sidebar.button("🔄 Reset Form",      on_click=reset_form)

# ═══════════════════════════════════════
# 7.  Collect data (no model yet)
# ═══════════════════════════════════════
if uploaded_file:
    raw_df, batch_mode = pd.read_csv(uploaded_file), True
else:
    raw_df, batch_mode = sidebar_inputs(), False

run_pred = st.sidebar.button("🚀 Run Prediction", use_container_width=True)
if not run_pred:
    st.stop()

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
# 9.  Batch or single prediction
# ═══════════════════════════════════════
if batch_mode:
    preds  = model.predict(X_enc)
    probs  = model.predict_proba(X_enc)[:, 1]
    raw_df["Prediction"]    = np.where(preds == 1, "Yes", "No")
    raw_df["Probability"]   = (probs*100).round(1).astype(str) + " %"
    raw_df["Risk Category"] = [label_risk(p) for p in probs]

    st.subheader("Batch Prediction Summary")
    st.dataframe(raw_df, use_container_width=True)
    st.info("Select a row below to inspect SHAP explanations.")
    sel = st.number_input("Row to inspect (1-based)",
                          min_value=1, max_value=len(raw_df), value=1)
    X_user  = X_enc.iloc[[sel-1]]
    user_df = raw_df.iloc[[sel-1]]
else:
    X_user, user_df = X_enc.iloc[[0]], raw_df.iloc[[0]]

# ═══════════════════════════════════════
# 10.  Show prediction
# ═══════════════════════════════════════
pred = model.predict(X_user)[0]
prob = model.predict_proba(X_user)[0, 1]
risk = label_risk(prob)

st.markdown("### Prediction Results")
st.markdown(
    f"""
<div style='border:2px solid #eee;border-radius:10px;padding:20px;background:#fafafa;'>
  <div style='display:flex;justify-content:space-between;font-size:18px;'>
    <div><b>Prediction</b><br><span style='font-size:24px'>{'Yes' if pred else 'No'}</span></div>
    <div><b>Probability</b><br><span style='font-size:24px'>{prob:.1%}</span></div>
    <div><b>Risk Category</b><br><span style='font-size:24px'>{risk}</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════
# 11.  SHAP explanations
# ═══════════════════════════════════════
st.subheader("🔍 SHAP Explanations")
sv = explainer.shap_values(X_user)
if isinstance(sv, list):
    sv = sv[1]

with st.expander("Global Impact — Beeswarm", expanded=True):
    fig, _ = plt.subplots()
    shap.summary_plot(sv, X_user, show=False)
    st.pyplot(fig); plt.clf()

with st.expander("Decision Path", expanded=False):
    fig, _ = plt.subplots()
    shap.decision_plot(explainer.expected_value, sv[0], X_user, show=False)
    st.pyplot(fig); plt.clf()

with st.expander("Local Force Plot", expanded=False):
    try:
        fig = shap.plots.force(explainer.expected_value, sv[0], X_user.iloc[0],
                               matplotlib=True, show=False)
        st.pyplot(fig)
    except Exception:
        st.warning("Force plot not supported – showing waterfall.")
        fig, _ = plt.subplots()
        shap.plots.waterfall(
            shap.Explanation(values=sv[0], base_values=explainer.expected_value,
                             data=X_user.iloc[0]),
            max_display=15, show=False)
        st.pyplot(fig)

with st.expander("Interactive Feature Impact", expanded=False):
    feat = st.selectbox("Choose feature", X_user.columns)
    idx  = X_user.columns.get_loc(feat)
    fig, _ = plt.subplots()
    shap.bar_plot(np.array([sv[0][idx]]), feature_names=[feat], show=False)
    st.pyplot(fig); plt.clf()

# ═══════════════════════════════════════
# 12.  History
# ═══════════════════════════════════════
record = user_df.copy()
record["Prediction"]    = "Yes" if pred else "No"
record["Probability"]   = f"{prob:.1%}"
record["Risk Category"] = risk
record["Timestamp"]     = datetime.now().strftime("%Y-%m-%d %H:%M")

st.session_state["history"] = pd.concat(
    [st.session_state["history"], record], ignore_index=True
)

st.subheader("Prediction History")
st.dataframe(st.session_state["history"], use_container_width=True)
if st.download_button("⬇️ Download", st.session_state["history"].to_csv(index=False),
                      file_name="prediction_history.csv"):
    st.toast("History downloaded!")

if st.button("🗑️ Clear History"):
    st.session_state["history"] = pd.DataFrame()
    st.rerun()
