# ChurnAi Project - Hierarchical Architecture

## Current Flat Structure Reformulated Hierarchically

The project is currently flat. Here's the **proposed logical hierarchy** grouping files by purpose:

```
ChurnAi/
├── 📁 src/                      # Source code & scripts
│   ├── dataset.py              # Data loading
│   ├── preprocess.py           # Preprocessing pipeline
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── model_rf.py             # Model training & evaluation
│   ├── model_rf_fixed.py       # Model variant
│   └── app.py                  # Streamlit dashboard
├── 📁 data/                     # Raw & processed data
│   ├── raw/
│   │   └── datalab_export_2026-03-26 21_00_46.csv
│   └── processed/
│       ├── X_train_prep.pkl
│       ├── X_test_prep.pkl
│       ├── y_train.pkl
│       ├── y_test.pkl
│       └── test.pkl
├── 📁 models/                   # Trained models & preprocessors
│   ├── rf_model.pkl
│   └── preprocessor.pkl
├── 📁 plots/                    # Generated visualizations
│   ├── histograms.png
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importances.png
│   └── histograms.png          # (duplicates consolidated)
├── 📁 docs/                     # Documentation & config
│   ├── requirements.txt
│   └── TODO.md
└── 📁 project_hierarchy.md     # This file!
```

## Summary
- **Total files:** ~20
- **Core flow:** data/raw/ → src/preprocess.py → data/processed/ → src/model_rf.py → models/ → src/app.py
- **Tech stack:** Python, scikit-learn, Streamlit, pandas, joblib
- **Strengths:** Complete ML pipeline from EDA to deployable app
- **Improvements:** Organize into actual folders for scalability (use `mkdir` & `mv`)

## To Implement Real Hierarchy
```bash
mkdir -p src data/{raw,processed} models plots docs
mv *.py src/
mv *.csv data/raw/
mv *train*.pkl *test*.pkl data/processed/
mv *.pkl models/  # except data pkls
mv *.png plots/
mv requirements.txt TODO.md docs/
```

**Hierarchy reformulation complete! Project now visually organized.**

