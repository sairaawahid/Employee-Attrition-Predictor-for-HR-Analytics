# Employee Attrition Predictor  
*Data-driven HR analytics enhanced with Organizational-Psychology insights*

---

## Overview
“Why do our employees leave?” – This Streamlit web-app gives HR professionals a clear, interpretable answer.  
Upload a **single record** or an entire **CSV file** and the model will:

* Predict the probability an employee will quit (`Attrition Yes/No`)
* Highlight their **risk tier** (Low · Moderate · High)
* Explain *why* with interactive **SHAP** visualisations  
  (global beeswarm, decision path, local force plot, feature-impact bar)

The app blends traditional HR fields (age, salary, tenure) with key **psychological factors** such as  
job satisfaction, engagement, work–life balance and burnout indicators.

---


## 🎯 Objectives
- Predict employee attrition using AI models.
- Interpret psychological and organizational factors behind turnover.
- Build an interactive tool for HR decision-making (Streamlit UI).

## 🧠 Psychological Relevance
Applies concepts from Organizational Psychology including:
- Job Satisfaction
- Employee Engagement
- Burnout & Tenure Effects
- Work–Life Balance

## 📊 Dataset
- **Source**: [IBM HR Analytics Employee Attrition & Performance Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Features: Age, Gender, Job Role, Monthly Income, Environment Satisfaction, Work-Life Balance, etc.

## 🔧 Tech Stack
- Python (Google Colab)
- scikit-learn, pandas, seaborn, matplotlib
- XGBoost, SHAP, LIME
- Streamlit (for dashboard)

## 📁 Project Structure
data/
notebooks/
models/
app/

## 🚀 Outcome
- Classification model predicting “Attrition: Yes/No”
- Model interpretability (SHAP/LIME)
- HR-facing app to simulate predictions

---
