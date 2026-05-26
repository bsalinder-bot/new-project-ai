import unittest
from app import app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SYMBIOTIC PLANETARY INTELLIGENCE', response.data)

    def test_predict_endpoint_text(self):
        response = self.client.post('/predict', json={'text': 'I need help'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('intent', data)
        self.assertIn('confidence', data)

    def test_predict_endpoint_signals(self):
        payload = {
            'signals': [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.05, 0.1, 0.15, 0.2, 0.25],
                [0.2, 0.25, 0.3, 0.35, 0.4],
                [0.1, 0.05, 0.0, -0.05, -0.1]
            ]
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(data['channels'], 4)
        self.assertEqual(data['time_steps'], 5)

    def test_predict_endpoint_invalid_json(self):
        response = self.client.post('/predict', data='not json', content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data['error'], 'Bad request')


if __name__ == '__main__':
    unittest.main()
