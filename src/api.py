import os
import sys
import joblib
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg

load_dotenv()

app = FastAPI(
    title="ChurnAi Prediction API",
    description="API pour prédictions de churn client avec clé API",
    version="1.0.0"
)

# Global model state
model = None
preprocessor = None
feature_names = cfg.FEATURE_NAMES

# Load models at startup
@app.on_event("startup")
async def load_models():
    global model, preprocessor
    try:
        model = joblib.load(cfg.MODEL_PATH)
        preprocessor = joblib.load(cfg.PREPROCESSOR_PATH)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e

# Pydantic model for input validation (13 features)
class PredictionInput(BaseModel):
    Call_Failure: float = Field(..., ge=0, description="Nombre d'échecs d'appels")
    Complaints: int = Field(..., ge=0, le=1, description="Nombre de plaintes")
    Subscription_Length: float = Field(..., ge=0, description="Durée abonnement (mois)")
    Charge_Amount: float = Field(..., ge=0, description="Montant facturé")
    Seconds_of_Use: float = Field(..., ge=0, description="Secondes d'utilisation")
    Frequency_of_use: float = Field(..., ge=0, description="Fréquence d'utilisation")
    Frequency_of_SMS: float = Field(..., ge=0, description="Fréquence SMS")
    Distinct_Called_Numbers: float = Field(..., ge=0, description="Numéros distincts appelés")
    Age_Group: int = Field(..., ge=1, le=5, description="Groupe d'âge (1-5)")
    Tariff_Plan: int = Field(..., ge=1, le=2, description="Plan tarifaire (1-2)")
    Status: int = Field(..., ge=1, le=2, description="Statut (1-2)")
    Age: float = Field(..., ge=15, description="Âge client")
    Customer_Value: float = Field(..., ge=0, description="Valeur client")

class PredictionOutput(BaseModel):
    prediction: int  # 0: No Churn, 1: Churn
    probability: float  # Proba churn (0-1)
    risk_level: str  # Low, Medium, High
    confidence: str
    explanation: List[str]

# API Key dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != cfg.API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")
    return x_api_key

@app.get("/health")
async def health_check():
    """Vérifier si l'API et les modèles sont opérationnels"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    return {"status": "OK", "service": "ChurnAi API", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
async def predict_churn(input_data: PredictionInput, api_key: str = Depends(verify_api_key)):
    """Prédire le risque de churn pour un client
    
    Exemple input:
    {
        "Call_Failure": 7.0,
        "Complaints": 0,
        "Subscription_Length": 38.0,
        "Charge_Amount": 100.0,
        ...
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    # Convert to DataFrame with exact column names
    import pandas as pd
    input_df = pd.DataFrame([input_data.dict()], columns=feature_names)
    
    # Ensure categorical dtypes
    cat_cols = ['Age Group', 'Tariff Plan', 'Status']
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)

    try:
        # Preprocess
        X_prep = preprocessor.transform(input_df)
        
        # Predict
        pred = model.predict(X_prep)[0]
        prob = model.predict_proba(X_prep)[0, 1]
        
        # Risk level
        if prob < 0.3:
            risk = "Low"
            conf = "Very Confident"
        elif prob < 0.7:
            risk = "Medium"
            conf = "Monitor Closely"
        else:
            risk = "High"
            conf = "Immediate Action Needed"
        
        # Top 3 feature contributions for explanation
        importances = model.feature_importances_
        top_indices = importances.argsort()[-3:][::-1]
        explanations = []
        for idx in top_indices:
            feat = feature_names[idx]
            imp_pct = importances[idx] * 100
            explanations.append(f"{feat}: {imp_pct:.1f}% influence")
        
        return PredictionOutput(
            prediction=int(pred),
            probability=round(float(prob), 4),
            risk_level=risk,
            confidence=conf,
            explanation=explanations
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.API_HOST, port=cfg.API_PORT)
