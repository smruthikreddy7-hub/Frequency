import unittest
import json
from app import create_app

class Phase1BTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_homepage_serves_chat_ui(self):
        """Verify the homepage serves the conversational UI."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Reasoning Console', response.data)
        self.assertIn(b'FREQUENCY', response.data)

    def test_get_models(self):
        """Verify GET /api/models returns model dictionary."""
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('models', data)
        self.assertIn('active_model', data)

    def test_chat_stream_endpoint(self):
        """Verify POST /api/chat/stream returns SSE stream with tokens."""
        payload = {
            "message": "Hi"
        }
        response = self.client.post(
            '/api/chat/stream',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith('text/event-stream'))
        
        # Read stream lines
        data_text = response.get_data(as_text=True)
        self.assertIn('data: ', data_text)
        self.assertIn('"done"', data_text)

if __name__ == '__main__':
    unittest.main()
