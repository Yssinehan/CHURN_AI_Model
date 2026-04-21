from flask import Flask, render_template, request, jsonify, session
import requests
import json
import re
from typing import Dict, Any
from config import ML_API_CONFIG, validate_input

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_in_production'

def analyze_message(message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Step 1 & 2: Analyze user message and structure data into JSON."""
    data = {}
    # Comma-separated positional (e.g., "12,homme,2,Internet,200,1500,1")
    if ',' in message and not ':' in message:
        field_names = list(ML_API_CONFIG["input_format"].keys())
        parts = [p.strip().strip('$') for p in message.split(',')]
        for i, part in enumerate(parts):
            if i < len(field_names):
                key = field_names[i]
                value = part
                # Clean/map
                if key == 'gender':
                    value = value.lower()
                    if 'homme' in value: value = 'male'
                    elif 'femme' in value: value = 'female'
                elif key == 'PhoneService':
                    value = 'Yes' if 'internet' not in value.lower() else 'No'
                data[key] = value
    # Key:value parsing
    if ':' in message:
        for pair in message.split(','):
            if ':' in pair:
                key, value = pair.strip().split(':', 1)
                data[key.strip().strip('$')] = value.strip()
    # NL patterns
    patterns = [
        (r'age\s+(\d+)', 'age'),
        (r'tenure\s+(\d+)', 'tenure'),
        (r'monthly\s*charges?\s+([\d.]+)', 'MonthlyCharges'),
        (r'total\s*charges?\s+([\d.]+)', 'TotalCharges'),
        (r'gender\s+(male|female|homme|femme)', 'gender'),
        (r'phone\s*service\s+(yes|no)', 'PhoneService'),
        (r'contract\s+(month-to-month|one year|two year)', 'Contract'),
    ]
    for pattern, key in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            data[key] = match.group(1)
    # Filter valid keys
    data = {k: v for k, v in data.items() if k in ML_API_CONFIG["input_format"]}
    
    if context:
        data.update(context)
    
    is_valid, msg = validate_input(data)
    if not is_valid:
        return {"error": msg}
    return {"data": data, "info": msg}

def call_ml_model(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: Call external ML API."""
    try:
        headers_template = ML_API_CONFIG["headers_template"]
        api_key = ML_API_CONFIG["api_key"]
        headers = {k: str(v).format(api_key=api_key) for k, v in headers_template.items()}
        response = requests.post(ML_API_CONFIG["endpoint"], json=structured["data"], headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def interpret_result(result: Dict[str, Any], info: str = "") -> str:
    """Step 4 & 5: Interpret and generate user-friendly response."""
    if "error" in result:
        return f"Error: {result['error']}. {info}"
    
    # Generic interpretation
    if "prediction" in result or "class" in result:
        pred = result.get("prediction") or result.get("class")
        prob = result.get("probability", 1.0)
        conf = "high" if prob > 0.8 else "medium" if prob > 0.5 else "low"
        if 'churn' in pred.lower():
            explanation = "Customer has high risk of churning. Consider retention actions."
        else:
            explanation = "Customer likely to stay. Good retention probability."
        return f"{info}\nThe model predicts: **{pred}** (confidence: {conf}, prob: {prob:.2%}). {explanation}"
    
    elif "probability" in result:
        prob = result["probability"]
        level = "high" if prob > 0.8 else "medium" if prob > 0.5 else "low"
        return f"{info}\nThe model estimates a probability of {prob:.2%}. This indicates a {level} level of likelihood."
    
    else:
        return f"{info}\nModel result: {json.dumps(result, indent=2)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    context = session.get('context', {})
    
    structured = analyze_message(user_message, context)
    model_result = call_ml_model(structured)
    response = interpret_result(model_result, structured.get('info', ''))
    
    session['context'] = structured["data"]
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

