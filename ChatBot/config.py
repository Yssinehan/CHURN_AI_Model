import os
from typing import Dict, Any

# Generic ML API configuration
# Adapt these to your specific API/model
ML_API_CONFIG = {
"endpoint": os.getenv("ML_API_ENDPOINT", "http://localhost:5000/predict"),
    "input_format": {  # Churn prediction features
        "age": "Customer age (int)",
        "gender": "gender (male/female)",
        "tenure": "tenure in months (int)",
        "PhoneService": "PhoneService (Yes/No)",
        "MonthlyCharges": "Monthly charges (float)",
        "TotalCharges": "Total charges (float)",
        "Contract": "Contract (Month-to-month/One year/Two year)",
    },
    "api_key": os.getenv("CHURNAI_API_KEY", "churnai_sk_0f8a2b4c6d7e9f1a3b5c8d0e2f4a6b8c1d3e5f7a9b0c2d4e6f8a0b2c4d6e8f0a"),
    "headers_template": {"Authorization": "Bearer {api_key}"},
    "output_types": ["classification", "probability", "score", "text"]  # Possible response types
}

def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """Generic validation for input data. Partial OK."""
    missing = [k for k in ML_API_CONFIG["input_format"] if k not in data]
    if len(data) == 0:
        return False, "No data extracted. Try 'age:35, gender:male' or 'age 35 tenure 12'"
    if missing:
        return True, f"Partial data OK ({len(data)}/{len(ML_API_CONFIG["input_format"])} fields). Missing: {', '.join(missing[:3])}..."
    return True, "All fields OK"

