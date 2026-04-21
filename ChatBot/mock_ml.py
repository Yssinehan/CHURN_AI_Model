from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    # Mock churn prediction
    tenure = float(data.get('tenure', 12))
    age = float(data.get('age', 30))
    charges = float(data.get('MonthlyCharges', 50)) + float(data.get('TotalCharges', 0))
    prob = min(0.95, 0.3 + (12 - tenure)/12 * 0.5 + (age/100) * 0.2 + (charges/1000)*0.1)
    pred = 'Churn' if prob > 0.5 else 'No Churn'
    return jsonify({'prediction': pred, 'probability': prob})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

