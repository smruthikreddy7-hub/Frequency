import os

class Config:
    """Application configuration for FREQUENCY."""
    
    # App Settings
    APP_NAME = "FREQUENCY"
    APP_VERSION = "1.4.0"
    # Disabling auto-reloader prevents watchdog from restarting Flask during streaming
    DEBUG = False
    HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
    PORT = int(os.environ.get("FLASK_PORT", 5000))
    SECRET_KEY = os.environ.get("SECRET_KEY", "frequency-cross-sense-secret-2026")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # Authentication + Admin defaults
    ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "AdminPass!123")
    LOGIN_MAX_ATTEMPTS = int(os.environ.get("LOGIN_MAX_ATTEMPTS", "5"))
    LOGIN_LOCKOUT_MINUTES = int(os.environ.get("LOGIN_LOCKOUT_MINUTES", "15"))
    DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "frequency.db"))
    
    # Ollama Local LLM Settings
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "qwen3:8b")
    OLLAMA_KEEP_ALIVE = "10m"
    
    # Token Budget & Context Window
    OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", 1024))
    OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", 4096))
    OLLAMA_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", 8))
    OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", 0.7))
    OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", 0.9))
    OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", 120))
