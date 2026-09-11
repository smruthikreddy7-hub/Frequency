import unittest
import uuid

from app import create_app


class AuthAndAdminTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_login_page_serves_html(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_signup_page_serves_html(self):
        response = self.client.get('/signup')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Create account', response.data)

    def test_user_can_sign_up(self):
        email = f"newuser_{uuid.uuid4().hex[:8]}@example.com"
        response = self.client.post(
            '/signup',
            data={
                'name': 'New User',
                'email': email,
                'password': 'SecurePass!123'
            },
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers.get('Location'), '/')

    def test_admin_demo_account_can_login(self):
        response = self.client.post(
            '/login',
            data={'email': 'admin@example.com', 'password': 'AdminPass!123'},
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers.get('Location'), '/admin')

    def test_invalid_credentials_return_generic_error(self):
        response = self.client.post(
            '/login',
            data={'email': 'admin@example.com', 'password': 'wrong-password'},
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn(b'Invalid email or password', response.data)

    def test_main_page_shows_logged_in_user(self):
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['user_name'] = 'System Administrator'
            sess['role'] = 'admin'

        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Logged in as', response.data)
        self.assertIn(b'System Administrator', response.data)

    def test_admin_dashboard_requires_admin_role(self):
        response = self.client.get('/admin', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/signin', response.headers.get('Location', ''))


if __name__ == '__main__':
    unittest.main()
