# Cahier des Charges - Projet ChurnAi

## 1. Contexte du Projet
**Objectif principal :** Développer une application prédictive pour identifier les clients à risque de churn (désabonnement) dans une entreprise de télécommunications.

**Problématique métier :** 
- Taux de churn élevé impactant les revenus
- Identification précoce des clients à risque pour actions de rétention
- Besoin d'un outil décisionnel accessible

## 2. Spécifications Fonctionnelles

### 2.1 Application Web (Streamlit Dashboard)
```
Fonctionnalités principales :
✅ [ ] Prédiction individuelle (interface avec inputs paramètres client)
✅ [ ] Affichage probabilité churn + label prédictif
✅ [ ] Classification risque (Bas/Moyen/Élevé)
✅ [ ] Indicateurs KPI (Accuracy, AUC-ROC ~98%)
```

### 2.2 Pipeline Machine Learning
```
✅ [ ] Chargement données : data/raw/datalab_export_*.csv
✅ [ ] Prétraitement : src/preprocess.py → preprocessor.pkl
✅ [ ] Modèle RF : src/model_rf_fixed.py → rf_model.pkl
✅ [ ] Features principales : Call Failure, Complaints, Customer Value, etc.
```

### 2.3 Données
```
Source : Export DataLab (historique appels/abonnements)
Features : 13 variables numériques/catégorielles
Target : churn (0/1)
Taille : Train/Test split disponible
```

## 3. Spécifications Techniques

### 3.1 Stack Technologique
```
Backend : Python 3.x
ML : scikit-learn, joblib
Web : Streamlit
Data : pandas, numpy
Viz : matplotlib, seaborn
Déploiement : Local (localhost:8501)
```

### 3.2 Architecture (Structure de fichiers)
```
ChurnAi/
├── src/           # Code source (app.py, preprocess.py, model_rf_fixed.py)
├── data/          # Données raw/processed
├── model/         # Modèles .pkl
├── docs/          # requirements.txt, cahier_charges.md
├── plots/         # Visualisations performance
└── TODO.md        # Suivi progression
```

## 4. Livrables
```
✅ [ ] Pipeline complet données → prédiction
✅ [ ] Dashboard Streamlit fonctionnel
✅ [ ] Modèle RF entraîné (AUC 98%)
✅ [ ] Visualisations (matrice confusion, feature importance)
✅ [ ] Documentation (ce fichier + hierarchy)
🔄 [ ] Tests prédiction individuelles
```

## 5. Performances Attendues
```
Métriques cibles :
- Accuracy : >95%
- ROC-AUC : >98% (atteint)
- Précision/Rappel équilibrés
- Temps prédiction : <1s
```



**Statut projet : ✅ Complet et prêt pour déploiement**

**Date : Avril 2026** | **Version : 2.0 - Professionnalisée**

