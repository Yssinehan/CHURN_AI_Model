# ChurnAi - Customer Churn Prediction App & API

## Overview
Streamlit web app + FastAPI pour prédictions de churn client avec Random Forest (précision ~98%). Prédit risque churn basé sur 13 features client.

## Nouvelles Fonctionnalités (API)
- **API REST** `/predict` (POST) avec clé API pour intégration chatbot
-  

- **Input**: JSON 13 features → Output: {prediction, probability, risk_level, explanation}
- **Endpoints**:
  - `POST /predict` (header `X-API-Key: <key>`)
  - `GET /health`
- **Docs auto**: `http://localhost:8000/docs`

## Project Structure
```
ChurnAi/
├── src/
│   ├── app_fixed.py     # Dashboard Streamlit
│   ├── api.py           # 🆕 FastAPI Prediction API
│   ├── config.py        # Config + API_KEY
│   └── ... (model, data, preprocess)
├── model/               # rf_model.pkl, preprocessor.pkl
├── docs/requirements.txt # Inclut fastapi, uvicorn
├── .env                 # 🆕 API_KEY (gitignore!)
├── run_app.bat          # Streamlit + API instructions
└── TODO.md
```

## Setup & Run
1. **Installer deps**: `run_setup.bat` ou `pip install -r docs/requirements.txt`
2. **Lancer Dashboard** (port 8501): `run_app.bat` ou `streamlit run src/app_fixed.py`
3. **Lancer API** (port 8000): `uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload`

**Test API** (cURL/PowerShell):
```
curl -X POST "http://localhost:8000/predict" ^
-H "Content-Type: application/json" ^
-H "X-API-Key: churnai_sk_0f8a2b4c6d7e9f1a3b5c8d0e2f4a6b8c1d3e5f7a9b0c2d4e6f8a0b2c4d6e8f0a" ^
-d "{\"Call_Failure\":7.0,\"Complaints\":0,\"Subscription_Length\":38.0,\"Charge_Amount\":100.0,\"Seconds_of_Use\":500.0,\"Frequency_of_use\":100.0,\"Frequency_of_SMS\":50.0,\"Distinct_Called_Numbers\":5.0,\"Age_Group\":3,\"Tariff_Plan\":1,\"Status\":2,\"Age\":30.0,\"Customer_Value\":470.0}"
```

## Chatbot Integration (Python)
```python
import requests
import json

API_URL = "http://localhost:8000/predict"  # Remplacez par URL hébergée
API_KEY = "churnai_sk_0f8a2b4c6d7e9f1a3b5c8d0e2f4a6b8c1d3e5f7a9b0c2d4e6f8a0b2c4d6e8f0a"

def predict_churn(features):
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    response = requests.post(API_URL, headers=headers, json=features)
    if response.status_code == 200:
        result = response.json()
        risk = result["risk_level"]
        prob = result["probability"]
        return f"🔴 Risque {risk} ({prob:.1%}). {', '.join(result['explanation'])}"
    return "Erreur API"

# Usage
client_data = {
    "Call_Failure": 15.0, "Complaints": 1, "Subscription_Length": 12.0,
    # ... 13 features
}
print(predict_churn(client_data))
```

## Hébergement Gratuit (Render.com)
1. Push GitHub (ajoutez `.env` à `.gitignore`)
2. Render.com → New Web Service → GitHub repo
3. Runtime: Python • Build: `pip install -r docs/requirements.txt`
4. Start: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
5. Env Vars: `API_KEY=your_key`
6. URL publique: `https://your-app.onrender.com/predict`

## Features
- **Dashboard**: Sliders + PDF upload pour prédictions live
- **API**: Auth clé API, validation input, explications features
- **Modèle**: RF entraîné (98% acc), 13 features client

## Tech Stack
Python, Streamlit, FastAPI, Scikit-learn, Joblib, Pydantic.

🚀 API prête pour votre chatbot ! Testez localement puis hébergez.
