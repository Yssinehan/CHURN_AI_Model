import logging
import re
from typing import Dict, Any, List
from pathlib import Path

from .recommendation import get_recommendation_json
from ..config import FEATURE_NAMES

logger = logging.getLogger(__name__)

class ChurnBot:
    """Intelligent chatbot for churn analysis."""
    
    def __init__(self):
        self.patterns = {
            r"(?:pourquoi|why).*?(?:churn|part|désabonne|quit)": self.explain_churn_risk,
            r"(?:recommand|action|que faire|what to do).*": self.get_recommendation,
            r"(?:risque|risk|probabilité|probability).*": self.get_risk_summary,
            r"(?:feature|important|facteur|cause).*": self.explain_features,
            r"hello|bonjour|hi": self.greet,
            r"(?:client|données|data).*": self.ask_for_client_data
        }
    
    def respond(self, message: str, client_data: Dict[str, Any] = None) -> str:
        """Generate response based on user message."""
        message_lower = message.lower().strip()
        
        # Route to specific handlers
        for pattern, handler in self.patterns.items():
            if re.search(pattern, message_lower):
                return handler(message, client_data)
        
        return self.default_response()
    
    def greet(self, message: str, client_data: Dict[str, Any] = None) -> str:
        return "Bonjour ! Je suis ChurnBot 🤖. Demandez-moi des analyses churn, recommandations ou explications !"
    
    def get_risk_summary(self, message: str, client_data: Dict[str, Any] = None) -> str:
        if not client_data:
            return "Veuillez fournir les données client pour l'analyse de risque."
        
        from .recommendation import predict_churn
        prob, pred = predict_churn(client_data)
        risk = "faible" if prob < 0.5 else "moyen" if prob < 0.75 else "élevé"
        
        return f"🔍 **Niveau de risque : {risk}** (Probabilité: {prob:.1%})\n\nFournissez plus de détails pour une recommandation personnalisée."
    
    def get_recommendation(self, message: str, client_data: Dict[str, Any] = None) -> str:
        if not client_data:
            return "📋 Pour une recommandation, donnez-moi les données client (ex: ancienneté, plaintes, etc.)"
        
        rec = get_recommendation_json(client_data)
        return f"""🎯 **Action Recommandée : {rec['recommended_action']}**

**Niveau :** {rec['risk_level']} ({rec['risk_probability']:.1%})
**Urgence :** {rec['urgency']}
**Justification :** {rec['justification']}

**Analyse :** {rec['analysis']}"""
    
    def explain_churn_risk(self, message: str, client_data: Dict[str, Any] = None) -> str:
        return """❓ **Pourquoi un client churn ?**

**Facteurs clés détectés par notre modèle :**
• 🔔 **Plaintes fréquentes** → Insatisfaction
• 📉 **Faible utilisation** → Désengagement  
• ⏳ **Ancienneté <12 mois** → Phase critique
• 💰 **Factures élevées** → Sensibilité prix

Fournissez des données client pour une analyse personnalisée !"""
    
    def explain_features(self, message: str, client_data: Dict[str, Any] = None) -> str:
        return """📊 **Facteurs de prédiction (Top 5 importance) :**
1. Customer Value (valeur client)
2. Call Failure (échecs appels)  
3. Complaints (réclamations)
4. Subscription Length (ancienneté)
5. Usage patterns (appels/SMS)

Le modèle analyse 13 features comportementales !"""
    
    def ask_for_client_data(self, message: str, client_data: Dict[str, Any] = None) -> str:
        features = ", ".join(FEATURE_NAMES[:5]) + "..."
        return f"👤 **Données client attendues :** {features}\n\nExemple: `{{'Subscription Length': 12, 'Complaints': 1}}`"
    
    def default_response(self) -> str:
        return """🤖 **ChurnBot prêt à aider !**

Demandes possibles :
• \"Quelle action pour ce client ?\"
• \"Niveau de risque ?\" 
• \"Pourquoi ce client va partir ?\"
• \"Facteurs importants ?\"

Fournissez les données client pour des réponses personnalisées !"""

# Global instance
bot = ChurnBot()

def get_chat_response(message: str, client_data: Dict[str, Any] = None) -> Dict[str, str]:
    """API-friendly chat response."""
    response = bot.respond(message, client_data)
    return {
        "message": response,
        "intent": "general"  # Could enhance with intent detection
    }

if __name__ == "__main__":
    bot = ChurnBot()
    print(bot.respond("Bonjour"))
    print(bot.respond("Quelle recommandation ?"))

