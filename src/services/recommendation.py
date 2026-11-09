import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..config import MODEL_PATH, PREPROCESSOR_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Structured recommendation output."""
    risk_level: str          # faible, moyen, élevé
    risk_prob: float         # 0-1
    analysis: str            # Detailed French analysis
    action: str              # ONE action: réduction forte, offre personnalisée, appel client
    justification: str       # Business rule explanation
    urgency: str             # Immédiat, Suivi, Surveillance

# Global model cache
_model = None
_preprocessor = None

def load_model():
    """Load model and preprocessor."""
    global _model, _preprocessor
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    return _model, _preprocessor

def predict_churn(client_data: Dict[str, Any]) -> Tuple[float, int]:
    """Predict churn probability."""
    model, preprocessor = load_model()
    
    # Validate and create input DataFrame
    input_df = pd.DataFrame([client_data])
    
    # Ensure categorical columns are int
    cat_cols = ["Age Group", "Tariff Plan", "Status"]
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)
    
    X_prep = preprocessor.transform(input_df)
    prob = model.predict_proba(X_prep)[0, 1]
    pred = model.predict(X_prep)[0]
    return prob, pred

def generate_recommendation(client_data: Dict[str, Any]) -> Recommendation:
    """Generate complete recommendation per business rules."""
    prob, pred = predict_churn(client_data)
    
    # Extract key metrics for analysis
    tenure = client_data.get('Subscription Length', 0)
    charges = client_data.get('Charge Amount', 0)
    complaints = client_data.get('Complaints', 0)
    usage = (client_data.get('Seconds of Use', 0) + client_data.get('Frequency of use', 0)) / 2
    contract = f"Plan {client_data.get('Tariff Plan', 1)} - Statut {client_data.get('Status', 1)}"
    
    # Risk level
    if prob < 0.5:
        risk_level = "faible"
        urgency = "Surveillance"
    elif prob < 0.75:
        risk_level = "moyen" 
        urgency = "Suivi"
    else:
        risk_level = "élevé"
        urgency = "Immédiat"
    
    # Reasons for risk
    reasons = []
    if complaints > 0: reasons.append("réclamations")
    if tenure < 12: reasons.append("faible ancienneté") 
    if usage < 600: reasons.append("basse utilisation")
    if charges > 300: reasons.append("dépenses élevées")
    
    analysis = f"Client: ancienneté {tenure} mois, dépense {charges:.0f}€, utilisation {usage:.0f}s/mois, {contract}."
    if reasons:
        analysis += f" Risques: {', '.join(reasons)}."
    else:
        analysis += " Profil stable mais ML détecte risque."
    
    # ONE action per rules
    if prob >= 0.75:
        action = "Réduction forte (-25%)"
        justification = "Churn >75%: réduction agressive immédiate"
    elif prob >= 0.50:
        action = "Offre personnalisée (bonus data)"
        justification = "Churn 50-75%: offre ciblée fidélisation"
    else:
        action = "Appel service client"
        justification = "Churn <50%: engagement préventif"
    
    return Recommendation(
        risk_level=risk_level,
        risk_prob=round(prob, 4),
        analysis=analysis,
        action=action,
        justification=justification,
        urgency=urgency
    )

def get_recommendation_json(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serializable recommendation."""
    rec = generate_recommendation(client_data)
    return {
        "risk_level": rec.risk_level,
        "risk_probability": rec.risk_prob,
        "analysis": rec.analysis,
        "recommended_action": rec.action,
        "justification": rec.justification,
        "urgency": rec.urgency
    }

# Sample usage
if __name__ == "__main__":
    sample_client = {
        "Call Failure": 15, "Complaints": 1, "Subscription Length": 6,
        "Charge Amount": 120, "Seconds of Use": 400, "Frequency of use": 80,
        "Frequency of SMS": 30, "Distinct Called Numbers": 8,
        "Age Group": 2, "Tariff Plan": 1, "Status": 1,
        "Age": 32, "Customer Value": 450
    }
    
    rec = generate_recommendation(sample_client)
    print(rec)
    print(get_recommendation_json(sample_client))

