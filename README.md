# Employee Attrition Predictor  

## Project Overview
“Why do our employees leave?” This Streamlit web-app gives HR professionals a clear, interpretable answer.  
Upload a **single record** or an entire **CSV file** and the model will:

* Predict the probability an employee will quit (`Attrition Yes/No`)
* Highlight their **risk tier** (Low · Moderate · High)
* Explain *why* with interactive **SHAP** visualisations  
  (global beeswarm, decision path, local force plot, feature-impact bar)

The app blends traditional HR fields (age, salary, tenure) with key **psychological factors** such as  
job satisfaction, engagement, work–life balance and burnout indicators.

---

## Objectives
- Predict employee attrition using machine learning techniques.
- Understand psychological and organizational drivers behind attrition.
- Offer explainable AI insights to support HR decision-making.
- Enable both individual predictions and batch analysis.
- Visualize results using SHAP force plots, beeswarm plots, and decision paths.

---

## Features
- **Real-time Attrition Prediction**: Predict whether an employee is at risk of leaving the organization.
- **Batch CSV Upload**: Upload a dataset of employees for multi-row predictions and exportable results.
- **Risk Categorization**: Displays probability-based risk levels — Low, Moderate, or High.
- **SHAP Explanations**: Understand which features drive each prediction (Force plot, Beeswarm, Decision plot).
- **Interactive Feature Impact**: Explore how individual features influence the prediction.
- **Download & Clear History**: Save prediction logs as CSV or clear them with one click.
- **Sample Data & Reset**: Test the app with sample inputs or reset the form instantly.
- **Psychology-Aligned Inputs**: Includes measures of job satisfaction, involvement, work-life balance, and more.

---

## Psychological Relevance
The app incorporates key concepts from Organizational Psychology:
- Job Satisfaction
- Burnout & Tenure Effects
- Employee Engagement
- Work–Life Balance
- Overtime & Performance Pressure

---

## Dataset
IBM HR Analytics Employee Attrition & Performance  
**Source** → <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>  
**Target**: Attrition (Yes/No)

---

## Tech Stack
- **Programming**: Python
- **Modeling**: scikit-learn, XGBoost
- **Visualization & Interpretation**: SHAP, matplotlib
- **Data Processing**: pandas, numpy
- **Web App**: Streamlit

---

## Usage
To use the app:

1. **Run locally with Streamlit:**
   ```bash
   streamlit run streamlit_app.py

2. **Inside the app:**
- Use the sidebar to input employee details or load sample data.
- Optionally upload a CSV file for batch prediction.
- Click Run Prediction to view risk probability and SHAP-based explanations.
- Use Interactive Feature Impact to understand the effect of each factor.
- Download or clear the prediction history as needed.

Inputs include dropdowns and sliders based on employee attributes. SHAP visualizations will explain each prediction transparently.

---

## Installation
```bash
# 1. Clone the repo
git clone https://github.com/<your-github-username>/Employee-Attrition-Predictor-for-HR-Analytics.git
cd Employee-Attrition-Predictor-for-HR-Analytics

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows use .venv\Scripts\activate

# 3. Install Python requirements
pip install -r requirements.txt

# 4. Launch the app
streamlit run streamlit_app.py
```

---

## Outcome
- Classification model predicting “Attrition: Yes/No”
- Model interpretability (SHAP)
- HR-facing app to simulate predictions

---

## Attribution
This project was developed by Sairaawahid.
If you use or adapt this app, please credit the author by linking to the original GitHub repository:
🔗 https://github.com/sairaawahid/Employee-Attrition-Predictor-for-HR-Analytics

---

## License
This project is licensed under the MIT License.
See the `LICENSE` file for more details.
