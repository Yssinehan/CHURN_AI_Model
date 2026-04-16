import logging
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns

from config import (
    DATA_RAW_PATH, DATA_PROCESSED_DIR, MODEL_DIR, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, RF_PARAM_GRID, MIN_ACCURACY, MIN_AUC, MIN_F1_CHURN
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load raw data with validation."""
    if not os.path.exists(DATA_RAW_PATH):
        raise FileNotFoundError(f"Raw data not found: {DATA_RAW_PATH}")
    df = pd.read_csv(DATA_RAW_PATH)
    logger.info(f"Loaded dataset shape: {df.shape}")
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not in columns")
    logger.info(f"Churn distribution:\n{df[TARGET].value_counts(normalize=True)}")
    return df.drop('index', axis=1, errors='ignore')

def preprocess_data(df):
    """Full preprocessing pipeline."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    # Validate features
    missing_num = set(NUMERICAL_FEATURES) - set(X.columns)
    missing_cat = set(CATEGORICAL_FEATURES) - set(X.columns)
    if missing_num or missing_cat:
        raise ValueError(f"Missing features: num={missing_num}, cat={missing_cat}")
    
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Save processed data
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(X_train_prep, os.path.join(DATA_PROCESSED_DIR, 'X_train_prep.pkl'))
    joblib.dump(X_test_prep, os.path.join(DATA_PROCESSED_DIR, 'X_test_prep.pkl'))
    joblib.dump(y_train, os.path.join(DATA_PROCESSED_DIR, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(DATA_PROCESSED_DIR, 'y_test.pkl'))
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    logger.info("Preprocessing complete and saved.")
    return X_train_prep, X_test_prep, y_train, y_test, preprocessor

def train_evaluate_model(X_train, X_test, y_train, y_test, preprocessor):
    """Train tuned RF with CV."""
    # Grid search
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rf, RF_PARAM_GRID, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    # Test predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'f1_churn': f1_score(y_test, y_pred, pos_label=1),
        'precision_churn': precision_score(y_test, y_pred, pos_label=1),
        'recall_churn': recall_score(y_test, y_pred, pos_label=1),
        'best_params': grid_search.best_params_
    }
    
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test AUC: {metrics['auc']:.4f}")
    logger.info(f"F1 Churn: {metrics['f1_churn']:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Save model & metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plots
    plot_model_figures(best_model, X_test, y_test, y_pred, preprocessor)
    
    return best_model, metrics

def plot_model_figures(model, X_test, y_test, y_pred, preprocessor):
    """Generate and save plots."""
    os.makedirs(os.path.join(MODEL_DIR, 'plots'), exist_ok=True)
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix - Tuned Random Forest')
    plt.savefig(os.path.join(MODEL_DIR, 'plots', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance with real names
    feature_names = NUMERICAL_FEATURES + [f"{cat}_cat" for cat in CATEGORICAL_FEATURES[1:]]  # Approximate
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(12,8))
    plt.title('Feature Importances - Top 15')
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f'feat_{i}' for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'plots', 'feature_importances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Plots saved.")

def main():
    """Main training pipeline."""
    try:
        df = load_data()
        X_train_prep, X_test_prep, y_train, y_test, preprocessor = preprocess_data(df)
        model, metrics = train_evaluate_model(X_train_prep, X_test_prep, y_train, y_test, preprocessor)
        
        # Validation checks
        assert metrics['accuracy'] >= MIN_ACCURACY, f"Accuracy {metrics['accuracy']:.4f} < {MIN_ACCURACY}"
        assert metrics['auc'] >= MIN_AUC, f"AUC {metrics['auc']:.4f} < {MIN_AUC}"
        assert metrics['f1_churn'] >= MIN_F1_CHURN, f"F1 Churn {metrics['f1_churn']:.4f} < {MIN_F1_CHURN}"
        
        logger.info("✅ Model training successful! All thresholds met.")
        logger.info(f"Model saved: {MODEL_PATH}")
        logger.info(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
