# Generic ML ChatBot

## Overview
A plug-and-play chatbot interface for any ML model via external API. Follows the exact specs: analyzes messages, structures data, calls API, interprets results generically.

## Quick Start
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Set API key (optional, fallback hardcoded):
   ```
   set CHURNAI_API_KEY=your_key  # Windows
   ```
3. Run:
   ```
   python app.py
   ```
4. Open http://localhost:5001

**Churn Example**: "Client age 35, gender male, tenure 12, MonthlyCharges 70.5, TotalCharges 800, PhoneService Yes, Contract Month-to-month"

## Features
- ChurnAI integrated (probability of churn).
- NL parsing: "age 35 tenure 5" works!
- Retention advice in responses.

## Features
- **Generic**: Works with classification, probability, scores, text.
- **Handles missing data**: Asks for required fields.
- **Multi-turn context**.
- **User-friendly responses**.

## Customization
- Edit `analyze_message()` for advanced NLP extraction.
- Update `interpret_result()` for model-specific logic.
- Supports any API returning JSON.

## ChurnAI API
Expects features: age, gender, tenure, PhoneService, MonthlyCharges, TotalCharges, Contract.

Response example:
```json
{"prediction": "Churn", "probability": 0.78}
```

**Security**: Use env var for production.
