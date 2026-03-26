import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
file_path = 'datalab_export_2026-03-26 21_00_46.csv'
df = pd.read_csv(file_path)

print('Dataset shape:', df.shape)
print('Target distribution:\\n', df['Churn'].value_counts(normalize=True))

# Drop useless 'index'
df = df.drop('index', axis=1)

# Define features
categorical_features = ['Age Group', 'Tariff Plan', 'Status']
numerical_features = ['Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
                      'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
                      'Distinct Called Numbers', 'Age', 'Customer Value']

X = df.drop('Churn', axis=1)
y = df['Churn']

print('X shape:', X.shape, 'y shape:', y.shape)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Train shapes:', X_train.shape, y_train.shape)
print('Test shapes:', X_test.shape, y_test.shape)

# Apply preprocessing
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print('Preprocessed train shape:', X_train_prep.shape)
print('Preprocessed test shape:', X_test_prep.shape)

# Save
joblib.dump(X_train_prep, 'X_train_prep.pkl')
joblib.dump(X_test_prep, 'X_test_prep.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print('\\nData preparation complete. Files saved: X_train_prep.pkl etc.')
print('Ready for modeling!')

