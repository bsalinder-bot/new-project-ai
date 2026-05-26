import unittest
import numpy as np
from model import SymbioticPlanetaryIntelligence, SimpleIntentModel


class TestSPIModel(unittest.TestCase):
    def setUp(self):
        self.spi = SymbioticPlanetaryIntelligence()
        self.text_model = SimpleIntentModel()

    def test_infer_returns_embedding(self):
        payload = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.25],
            [0.2, 0.25, 0.3, 0.35, 0.4],
            [0.1, 0.05, 0.0, -0.05, -0.1]
        ]
        result = self.spi.infer(payload)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['channels'], 4)
        self.assertEqual(result['time_steps'], 5)
        self.assertEqual(result['shape'], [1, 256])
        self.assertEqual(len(result['embedding']), 256)

    def test_infer_requires_equal_channel_length(self):
        with self.assertRaises(ValueError):
            self.spi.infer([[0.1, 0.2], [0.1]])

    def test_text_model_fallback(self):
        result = self.text_model.predict('I need help')
        self.assertIn('intent', result)
        self.assertIn('confidence', result)
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)

    def test_text_model_empty_returns_unknown(self):
        result = self.text_model.predict('')
        self.assertEqual(result['intent'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)


if __name__ == '__main__':
    unittest.main()
