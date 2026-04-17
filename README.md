# ChurnAi - Customer Churn Prediction App

## Overview
Streamlit web app for predicting customer churn using Random Forest model. Features EDA, preprocessing, model training, and predictions.

## Project Structure
```
ChurnAi/
├── src/              # Source code
│   ├── app.py        # Main Streamlit app
│   ├── dataset.py    # Data loading
│   ├── preprocess.py # Preprocessing pipeline
│   ├── model_rf_fixed.py # RF model training/prediction
│   ├── eda.py        # Exploratory Data Analysis
│   └── dataset.py
├── data/             # Raw/processed data (ignored in Git)
├── model/            # Trained models (ignored)
├── plots/            # Generated plots (ignored)
├── docs/             # Requirements, specs
│   ├── requirements.txt
│   └── cahier_charges.md
├── TODO.md           # Tasks
└── project_hierarchy.md
```

## Setup & Run (Fixed Plotly Error)
**Quick Start (run_setup.bat created):**
1. New terminal: `.\\run_setup.bat` (activates venv + installs plotly/streamlit)
2. Verify: `python -c "import plotly; print('OK')"`
3. Run app: `.\\run_app.bat` OR `streamlit run src/app_fixed.py`

**Full Setup:**
1. venv exists - activate: `venv\Scripts\activate.bat`
2. `pip install -r docs/requirements.txt`
3. `streamlit run src/app_fixed.py` (localhost:8501)

Models ready in `model/`. Test PDF upload + live prediction!

## Features
- Interactive EDA with histograms, feature importance.
- Train RF model, view confusion matrix.
- Predict churn on new data.

## Tech Stack
- Python, Streamlit, Pandas, Scikit-learn, Plotly, Joblib.

Built by BLACKBOXAI.
