import unittest
from app import app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Simple AI Intent Demo', response.data)

    def test_predict_endpoint_empty(self):
        response = self.client.post('/predict', json={'text': ''})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['intent'], 'unknown')
        self.assertEqual(data['confidence'], 0.0)


if __name__ == '__main__':
    unittest.main()
