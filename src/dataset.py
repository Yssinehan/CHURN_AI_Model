import pandas as pd

# Chemin vers ton fichier CSV
file_path = "datalab_export_2026-03-26 21_00_46.csv"

# Lire le fichier
df = pd.read_csv(file_path)

# Vérifier les 5 premières lignes
print(df.head())

# Vérifier la taille du dataset et les colonnes
print(df.shape)
print(df.columns)

# Infos générales
print(df.info())

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Compter la cible (churn)
print(df['Churn'].value_counts())