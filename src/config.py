"""Configuration for ChurnAi ML Pipeline"""

import os

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "datalab_export_2026-03-26 21_00_46.csv")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Ensure directories
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature columns (exact names from dataset)
NUMERICAL_FEATURES = [
    'Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
    'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
    'Distinct Called Numbers', 'Age', 'Customer Value'
]
CATEGORICAL_FEATURES = ['Age Group', 'Tariff Plan', 'Status']
TARGET = 'Churn'

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Model hyperparameters grid
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Performance thresholds for tests
MIN_ACCURACY = 0.85
MIN_AUC = 0.90
MIN_F1_CHURN = 0.80

# Feature names for upload parsing
FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# API Configuration
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY not set in .env file")

API_HOST = "0.0.0.0"
API_PORT = 8000
