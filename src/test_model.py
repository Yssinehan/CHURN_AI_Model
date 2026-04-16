"""Unit tests for model pipeline."""

import unittest
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import pytest  # or unittest

from config import (
    MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, MIN_ACCURACY, MIN_AUC, MIN_F1_CHURN,
    DATA_RAW_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
)
from model import load_data, preprocess_data, train_evaluate_model

class TestModelPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Run full pipeline before tests."""
        print("Running full training pipeline for tests...")
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        cls.model, cls.metrics = train_evaluate_model(X_train, X_test, y_train, y_test, preprocessor)
        cls.X_test_prep = joblib.load(os.path.join(DATA_PROCESSED_DIR, 'X_test_prep.pkl'))
        cls.y_test = joblib.load(os.path.join(DATA_PROCESSED_DIR, 'y_test.pkl'))
    
    def test_model_files_exist(self):
        """Test saved artifacts."""
        self.assertTrue(os.path.exists(MODEL_PATH), "Model not saved")
        self.assertTrue(os.path.exists(PREPROCESSOR_PATH), "Preprocessor not saved")
        self.assertTrue(os.path.exists(METRICS_PATH), "Metrics not saved")
    
    def test_performance_thresholds(self):
        """Test metrics meet thresholds."""
        metrics = self.metrics
        self.assertGreaterEqual(metrics['accuracy'], MIN_ACCURACY, f"Accuracy too low: {metrics['accuracy']}")
        self.assertGreaterEqual(metrics['auc'], MIN_AUC, f"AUC too low: {metrics['auc']}")
        self.assertGreaterEqual(metrics['f1_churn'], MIN_F1_CHURN, f"F1 churn too low: {metrics['f1_churn']}")
        print(f"✅ Metrics OK: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, F1={metrics['f1_churn']:.4f}")
    
    def test_model_predictions(self):
        """Test model predicts valid probabilities."""
        probs = self.model.predict_proba(self.X_test_prep)[:, 1]
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)), "Invalid probabilities")
        preds = self.model.predict(self.X_test_prep)
        acc = accuracy_score(self.y_test, preds)
        self.assertGreater(acc, 0.8, f"Test accuracy too low: {acc}")
    
    def test_feature_importance(self):
        """Test feature importances sum to 1."""
        importances = self.model.feature_importances_
        np.testing.assert_almost_equal(importances.sum(), 1.0, decimal=5)
    
    def test_preprocessor_transform(self):
        """Test preprocessor shape consistency."""
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        # Load sample raw data
        df = pd.read_csv(DATA_RAW_PATH).head(5)
        X_sample = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        X_transformed = preprocessor.transform(X_sample)
        self.assertEqual(X_transformed.shape[1], 13, "Wrong transformed shape")  # 10 num + 2 cat

def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestModelPipeline))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All model tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")

if __name__ == "__main__":
    main()
