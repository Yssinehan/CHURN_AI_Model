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

## Setup & Run
1. Clone repo: `git clone https://github.com/Yssinehan/ChurnAi.git`
2. Create venv: `python -m venv venv`
3. Activate: `venv\\Scripts\\activate` (Windows)
4. Install deps: `pip install -r docs/requirements.txt`
5. Place data CSV in `data/raw/`
6. Run app: `streamlit run src/app.py`

## Features
- Interactive EDA with histograms, feature importance.
- Train RF model, view confusion matrix.
- Predict churn on new data.

## Tech Stack
- Python, Streamlit, Pandas, Scikit-learn, Plotly, Joblib.

Built by BLACKBOXAI.
