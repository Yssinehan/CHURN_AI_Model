import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = "datalab_export_2026-03-26 21_00_46.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDataset info:")
print(df.info())

# Drop 'index' column as it's not useful
df = df.drop('index', axis=1)

# 1. Univariate distributions (histograms for numerical features)
numerical_cols = ['Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount', 
                  'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 
                  'Distinct Called Numbers', 'Age', 'Customer Value']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
    axes[idx].set_title(col)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('histograms.png')
plt.show()

# 2. Boxplots of key features by Churn
key_features = ['Call Failure', 'Complaints', 'Subscription Length', 'Customer Value']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(key_features):
    row = idx // 2
    col = idx % 2
    sns.boxplot(data=df, x='Churn', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Churn')

plt.tight_layout()
plt.savefig('boxplots_churn.png')
plt.show()

# 3. Correlation heatmap (numerical features)
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# 4. Churn rate by categorical features
categorical_cols = ['Age Group', 'Tariff Plan', 'Status']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, col in enumerate(categorical_cols):
    churn_rate = df.groupby(col)['Churn'].mean()
    axes[idx].bar(churn_rate.index, churn_rate.values)
    axes[idx].set_title(f'Churn Rate by {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Churn Rate')

plt.tight_layout()
plt.savefig('churn_by_category.png')
plt.show()

# 5. Insights
print("\n=== EDA Insights ===")
print("Churn distribution:")
print(df['Churn'].value_counts(normalize=True))

print("\nTop correlations with Churn:")
corr_churn = df[numerical_cols + ['Churn']].corr()['Churn'].sort_values(ascending=False)
print(corr_churn[1:11])  # Top positive

print("\nHigh value customers churn rate:")
print(df.groupby(pd.cut(df['Customer Value'], bins=5))['Churn'].mean())

print("EDA complete. Check saved PNG files: histograms.png, boxplots_churn.png, correlation_heatmap.png, churn_by_category.png")

