import sqlite3
import os
import json
from datetime import datetime, date, timezone, timedelta
from typing import Dict, Any, List, Optional, Union

from werkzeug.security import check_password_hash, generate_password_hash

from config import Config

DB_FILE = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "frequency.db"))

class Database:
    """
    Local SQLite persistence engine for FREQUENCY.
    Stores daily lifestyle context, sensory signal sessions, and historical baselines.
    """

    @classmethod
    def get_connection(cls) -> sqlite3.Connection:
        """Create and return database connection."""
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def init_db(cls):
        """Initialize required database tables."""
        with cls.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    entry_type TEXT NOT NULL DEFAULT 'general',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            # 1. Daily Context Table (Phase 4)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    date TEXT NOT NULL,
                    sleep_hours REAL DEFAULT 7.5,
                    sleep_quality TEXT DEFAULT 'moderate',
                    workload TEXT DEFAULT 'normal',
                    activity_level TEXT DEFAULT 'moderate',
                    mood_score INTEGER DEFAULT 7,
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, date)
                )
            """)

            # 2. Session Signals History (Phase 5 Baseline data source)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    sentiment TEXT,
                    emotional_valence REAL,
                    mean_pitch_hz REAL,
                    vocal_energy_rms REAL,
                    speech_rate_wpm REAL,
                    pause_duration_ratio REAL,
                    cognitive_load TEXT,
                    raw_signals_json TEXT
                )
            """)

            cls._ensure_demo_admin(cursor)

            # Insert sample historical context records if empty (to establish a rich 7-day baseline)
            cursor.execute("SELECT COUNT(*) as count FROM daily_context")
            if cursor.fetchone()["count"] == 0:
                cls._seed_default_history(cursor)

            cls._seed_demo_entries(cursor)
            conn.commit()

    @classmethod
    def _ensure_demo_admin(cls, cursor: sqlite3.Cursor):
        """Create the default admin account if it does not already exist."""
        admin_email = (Config.ADMIN_EMAIL or "admin@example.com").strip().lower()
        admin_name = "System Administrator"
        existing = cursor.execute(
            "SELECT id FROM users WHERE email = ?",
            (admin_email,)
        ).fetchone()
        if existing:
            cursor.execute(
                "UPDATE users SET role = 'admin', name = ?, status = 'active' WHERE email = ?",
                (admin_name, admin_email)
            )
            return

        password_hash = generate_password_hash(Config.ADMIN_PASSWORD, method="pbkdf2:sha256")
        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, role, created_at, last_login_at, status)
            VALUES (?, ?, ?, 'admin', ?, NULL, 'active')
            """,
            (admin_name, admin_email, password_hash, datetime.now(timezone.utc).isoformat())
        )

    @classmethod
    def _seed_demo_entries(cls, cursor: sqlite3.Cursor):
        """Seed entry data for admin dashboard examples."""
        existing_count = cursor.execute("SELECT COUNT(*) as count FROM entries").fetchone()["count"]
        if existing_count > 0:
            return

        admin_user = cursor.execute(
            "SELECT id FROM users WHERE email = ?",
            ((Config.ADMIN_EMAIL or "admin@example.com").strip().lower(),)
        ).fetchone()
        if not admin_user:
            return
        admin_id = admin_user["id"]
        now = datetime.now(timezone.utc)
        sample_entries = [
            (admin_id, "Morning check-in", "Sleep quality was strong and the plan for today is clear.", "wellbeing", {"source": "web"}),
            (admin_id, "Customer feedback", "Users want clearer onboarding and faster status updates.", "feedback", {"source": "support"}),
            (admin_id, "Product note", "The reasoning console is more consistent when context is logged before prompting.", "note", {"source": "product"}),
            (admin_id, "Research insight", "Cross-sense alignments track best when sleep and workload are recorded daily.", "research", {"source": "analysis"})
        ]
        for user_id, title, content, entry_type, metadata in sample_entries:
            cursor.execute(
                """
                INSERT INTO entries (user_id, title, content, entry_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, title, content, entry_type, json.dumps(metadata), (now - timedelta(days=len(sample_entries) - 1)).isoformat())
            )

    @classmethod
    def create_user(cls, name: str, email: str, password: str, role: str = "user") -> Dict[str, Any]:
        """Create a new application user with a hashed password."""
        normalized_email = (email or "").strip().lower()
        name_str = (name or "").strip()
        if not normalized_email or not name_str or not password:
            raise ValueError("Name, email, and password are required.")

        with cls.get_connection() as conn:
            cursor = conn.cursor()
            existing = cursor.execute("SELECT id FROM users WHERE email = ?", (normalized_email,)).fetchone()
            if existing:
                raise ValueError("A user with that email already exists.")

            password_hash = generate_password_hash(password, method="pbkdf2:sha256")
            now_str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash, role, created_at, last_login_at, status)
                VALUES (?, ?, ?, ?, ?, NULL, 'active')
                """,
                (name_str, normalized_email, password_hash, role, now_str)
            )
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
            return dict(user) if user else None

    @classmethod
    def get_user_by_id(cls, user_id: int) -> Optional[Dict[str, Any]]:
        with cls.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    @classmethod
    def get_user_by_email(cls, email: str) -> Optional[Dict[str, Any]]:
        normalized_email = (email or "").strip().lower()
        if not normalized_email:
            return None
        with cls.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE LOWER(email) = ?", (normalized_email,)).fetchone()
            return dict(row) if row else None

    @classmethod
    def authenticate_user(cls, email: str, password: str) -> Optional[Dict[str, Any]]:
        user = cls.get_user_by_email(email)
        if not user:
            return None
        if not check_password_hash(user["password_hash"], password):
            return None
        cls.update_last_login(user["id"])
        return cls.get_user_by_id(user["id"])

    @classmethod
    def update_last_login(cls, user_id: int):
        with cls.get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), user_id)
            )
            conn.commit()

    @classmethod
    def list_users(cls) -> List[Dict[str, Any]]:
        with cls.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, name, email, role, created_at, last_login_at,
                       CASE WHEN last_login_at IS NOT NULL AND datetime(last_login_at) >= datetime('now', '-30 days') THEN 'active' ELSE 'inactive' END AS status
                FROM users
                ORDER BY created_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    @classmethod
    def list_entries(cls) -> List[Dict[str, Any]]:
        with cls.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.user_id, u.name as user_name, u.email as user_email, e.title, e.content, e.entry_type, e.metadata_json, e.created_at
                FROM entries e
                JOIN users u ON u.id = e.user_id
                ORDER BY e.created_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    @classmethod
    def create_entry(cls, user_id: int, title: str, content: str, entry_type: str = "general", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                INSERT INTO entries (user_id, title, content, entry_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, title[:120], content[:2000], entry_type, json.dumps(metadata or {}), now_str)
            )
            entry_id = cursor.lastrowid
            conn.commit()
            entry = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            return dict(entry)

    @classmethod
    def get_dashboard_stats(cls) -> Dict[str, Any]:
        with cls.get_connection() as conn:
            total_users = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()["count"]
            total_entries = conn.execute("SELECT COUNT(*) as count FROM entries").fetchone()["count"]
            active_7 = conn.execute(
                "SELECT COUNT(*) as count FROM users WHERE last_login_at IS NOT NULL AND datetime(last_login_at) >= datetime('now', '-7 days')"
            ).fetchone()["count"]
            active_30 = conn.execute(
                "SELECT COUNT(*) as count FROM users WHERE last_login_at IS NOT NULL AND datetime(last_login_at) >= datetime('now', '-30 days')"
            ).fetchone()["count"]
            signups_series = conn.execute(
                """
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM users
                GROUP BY DATE(created_at)
                ORDER BY date ASC
                LIMIT 30
                """
            ).fetchall()

            return {
                "total_users": total_users,
                "total_entries": total_entries,
                "active_7_days": active_7,
                "active_30_days": active_30,
                "signups_by_day": [dict(row) for row in signups_series],
            }

    @classmethod
    def _seed_default_history(cls, cursor: sqlite3.Cursor):
        """Seed a realistic 7-day historical record for baseline calculations."""
        sample_days = [
            ("2026-08-16", 7.8, "restful", "normal", "active", 8, 0.038, 145.0, 130.0, 0.35),
            ("2026-08-17", 7.2, "moderate", "normal", "moderate", 7, 0.035, 150.0, 128.0, 0.20),
            ("2026-08-18", 6.8, "moderate", "heavy", "moderate", 6, 0.032, 155.0, 132.0, -0.10),
            ("2026-08-19", 7.5, "restful", "normal", "moderate", 8, 0.039, 142.0, 126.0, 0.40),
            ("2026-08-20", 8.0, "restful", "light", "active", 9, 0.042, 140.0, 125.0, 0.50),
            ("2026-08-21", 7.0, "moderate", "normal", "moderate", 7, 0.036, 148.0, 129.0, 0.15),
        ]
        now_str = datetime.now(timezone.utc).isoformat()
        for d, sleep, qual, work, act, mood, rms, wpm, pitch, val in sample_days:
            cursor.execute("""
                INSERT OR IGNORE INTO daily_context 
                (user_id, date, sleep_hours, sleep_quality, workload, activity_level, mood_score, notes, created_at, updated_at)
                VALUES ('default_user', ?, ?, ?, ?, ?, ?, 'Historical baseline log', ?, ?)
            """, (d, sleep, qual, work, act, mood, now_str, now_str))

            cursor.execute("""
                INSERT INTO session_history 
                (session_id, user_id, timestamp, prompt, modality, sentiment, emotional_valence, mean_pitch_hz, vocal_energy_rms, speech_rate_wpm, pause_duration_ratio, cognitive_load, raw_signals_json)
                VALUES (?, 'default_user', ?, 'Historical entry', 'multimodal', ?, ?, ?, ?, ?, 0.18, 'moderate', '{}')
            """, (f"seed_{d}", f"{d}T12:00:00Z", "positive" if val > 0 else "neutral", val, pitch, rms, wpm))

    @classmethod
    def save_context(
        cls,
        user_id: str = "default_user",
        sleep_hours: float = 7.5,
        sleep_quality: str = "moderate",
        workload: str = "normal",
        activity_level: str = "moderate",
        mood_score: int = 7,
        notes: str = "",
        target_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save or update today's context."""
        day_str = target_date or date.today().isoformat()
        now_str = datetime.now(timezone.utc).isoformat()

        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO daily_context 
                (user_id, date, sleep_hours, sleep_quality, workload, activity_level, mood_score, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    sleep_hours=excluded.sleep_hours,
                    sleep_quality=excluded.sleep_quality,
                    workload=excluded.workload,
                    activity_level=excluded.activity_level,
                    mood_score=excluded.mood_score,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
            """, (user_id, day_str, sleep_hours, sleep_quality, workload, activity_level, mood_score, notes, now_str, now_str))
            conn.commit()

        return cls.get_context(user_id, day_str)

    @classmethod
    def get_context(cls, user_id: str = "default_user", target_date: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve daily context for a given date."""
        day_str = target_date or date.today().isoformat()
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_context WHERE user_id = ? AND date = ?
            """, (user_id, day_str))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return {
            "user_id": user_id,
            "date": day_str,
            "sleep_hours": 7.0,
            "sleep_quality": "moderate",
            "workload": "normal",
            "activity_level": "moderate",
            "mood_score": 7,
            "notes": ""
        }

    @classmethod
    def log_session(
        cls,
        session_id: str,
        prompt: str,
        modality: str,
        text_signals: Optional[Union[Dict[str, Any], str]] = None,
        voice_signals: Optional[Dict[str, Any]] = None,
        user_id: str = "default_user"
    ):
        """Record session telemetry to SQLite for baseline history."""
        now_str = datetime.now(timezone.utc).isoformat()
        
        t_dict = text_signals if isinstance(text_signals, dict) else {}
        v_dict = voice_signals if isinstance(voice_signals, dict) else {}

        sentiment = t_dict.get("sentiment")
        valence = t_dict.get("emotional_valence")
        cog_load = t_dict.get("cognitive_load")
        
        pitch = v_dict.get("mean_pitch_hz")
        energy = v_dict.get("vocal_energy_rms")
        speech_rate = v_dict.get("speech_rate_wpm")
        pause = v_dict.get("pause_duration_ratio")

        raw_signals = {
            "text": text_signals if isinstance(text_signals, dict) else {"raw": str(text_signals)},
            "voice": voice_signals
        }

        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_history 
                (session_id, user_id, timestamp, prompt, modality, sentiment, emotional_valence, mean_pitch_hz, vocal_energy_rms, speech_rate_wpm, pause_duration_ratio, cognitive_load, raw_signals_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, now_str, prompt[:200], modality, sentiment, valence, pitch, energy, speech_rate, pause, cog_load, json.dumps(raw_signals)))
            conn.commit()

    @classmethod
    def get_historical_context(cls, user_id: str = "default_user", limit: int = 14) -> List[Dict[str, Any]]:
        """Get past daily context entries."""
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_context WHERE user_id = ? ORDER BY date DESC LIMIT ?
            """, (user_id, limit))
            return [dict(r) for r in cursor.fetchall()]

    @classmethod
    def get_historical_sessions(cls, user_id: str = "default_user", limit: int = 30) -> List[Dict[str, Any]]:
        """Get past session records."""
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM session_history WHERE user_id = ? ORDER BY id DESC LIMIT ?
            """, (user_id, limit))
            return [dict(r) for r in cursor.fetchall()]

    @classmethod
    def get_7day_trends(cls, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Get synthesized 7-day multi-modal trend records for the dashboard."""
        contexts = cls.get_historical_context(user_id=user_id, limit=7)
        contexts.reverse()  # Chronological order

        trend_points = []
        for c in contexts:
            d_str = c["date"]
            sleep = c["sleep_hours"]
            mood = c["mood_score"]
            workload = c["workload"]
            
            # Simple composite estimate for trend point
            energy = min(100.0, max(20.0, (sleep / 7.5) * 60.0 + (mood / 10.0) * 40.0))
            focus = 85.0 if workload == "light" else (75.0 if workload == "normal" else 55.0)

            trend_points.append({
                "date": d_str,
                "sleep_hours": sleep,
                "mood_score": mood,
                "workload": workload,
                "energy_index": round(energy, 1),
                "focus_index": round(focus, 1)
            })

        return trend_points

# Auto-initialize database on import
Database.init_db()
