import os, unittest
from unittest.mock import patch
os.environ["API_KEY"] = "local-test-placeholder"
from src.services.chatbot import ChurnBot
from src.services import recommendation as rec

class ServiceTests(unittest.TestCase):
    def test_greeting(self):
        self.assertIn("Bonjour", ChurnBot().respond("Bonjour"))
    def test_missing_client(self):
        self.assertIn("données client", ChurnBot().respond("risque"))
    def test_feature_prompt_formats(self):
        self.assertIn("'Subscription Length': 12", ChurnBot().ask_for_client_data("client"))
    def test_recommendation_thresholds(self):
        for prob, level, action in [(0.49, "faible", "Appel"), (0.5, "moyen", "Offre"), (0.75, "élevé", "Réduction"), (0.9, "élevé", "Réduction")]:
            with self.subTest(prob=prob), patch.object(rec, "predict_churn", return_value=(prob, int(prob >= .5))):
                result = rec.get_recommendation_json({"Subscription Length": 24})
                self.assertEqual(result["risk_level"], level)
                self.assertTrue(result["recommended_action"].startswith(action))
                self.assertEqual(result["risk_probability"], prob)
if __name__ == "__main__": unittest.main()
