import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

print('Loading preprocessed data...')
X_train = joblib.load('X_train_prep.pkl')
X_test = joblib.load('X_test_prep.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

print('Loaded shapes:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)

print('Model trained. Saving rf_model.pkl early...')
joblib.dump(rf, 'rf_model.pkl')
print('rf_model.pkl saved!')

# Predict
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC-AUC: {auc:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('rf_confusion_matrix.png')
plt.close()

# Feature Importances
n_features = X_train.shape[1]
feature_names = [f'feature_{i}' for i in range(n_features)]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,8))
plt.title('Feature Importances - Random Forest')
plt.bar(range(n_features), importances[indices])
plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('rf_feature_importances.png')
plt.close()

print('\nTop 5 features:')
for i in range(min(5, len(indices))):
    print(f'{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}')

print('Model training & plots complete. rf_model.pkl ready for app.py!')

