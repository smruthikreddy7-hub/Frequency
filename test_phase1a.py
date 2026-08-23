import unittest
from app import create_app

class Phase1ATestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_homepage_route(self):
        """Verify the homepage serves HTML with 200 OK."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'FREQUENCY', response.data)
        self.assertIn(b'Cross-Sense', response.data)

    def test_status_endpoint(self):
        """Verify GET /api/status returns valid operational JSON structure."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        
        data = response.get_json()
        self.assertEqual(data.get('status'), 'operational')
        self.assertEqual(data.get('app_name'), 'FREQUENCY')
        self.assertIn('version', data)
        self.assertIn('timestamp', data)
        self.assertIn('reasoning_backend', data)
        self.assertIn('active_model', data['reasoning_backend'])

if __name__ == '__main__':
    unittest.main()
