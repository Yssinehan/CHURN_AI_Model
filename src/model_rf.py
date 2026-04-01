import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load preprocessed data
X_train = joblib.load('X_train_prep.pkl')
X_test = joblib.load('X_test_prep.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

print('Loaded shapes:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# Random Forest with balanced weights for imbalance
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Train
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC-AUC: {auc:.4f}')
print('\\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('rf_confusion_matrix.png')
plt.show()

# Feature Importances
feature_names = [f'feat_{i}' for i in range(16)]  # Preprocessed 16 features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title('Feature Importances - Random Forest')
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('rf_feature_importances.png')
plt.show()

print('\\nTop 5 features:')
for i in range(5):
    print(f'{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}')

# Save model
joblib.dump(rf, 'rf_model.pkl')
print('\\nModel saved as rf_model.pkl')
print('Baseline Random Forest complete!')

