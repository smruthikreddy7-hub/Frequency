from datetime import datetime, timezone, timedelta
import json
import logging
import re
from functools import wraps
from collections import defaultdict

from flask import Flask, jsonify, render_template, request, Response, stream_with_context, redirect, url_for, session

from config import Config
from llm_client import LLMClient
from schema import FrequencyInput, TextSignalData, VoiceSignalData, ContextData, BaselineData, FusionData
from prompt_builder import PromptBuilder
from text_analyzer import TextAnalyzer
from voice_analyzer import VoiceAnalyzer
from database import Database
from baseline_engine import BaselineEngine
from fusion_engine import FusionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("frequency")

login_attempts = defaultdict(list)


def sanitize_text(value, max_length=2000):
    """Normalize and sanitize raw user input for display/storage."""
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text[:max_length]


def rate_limited(email):
    """Basic in-memory login throttling to discourage brute force attempts."""
    normalized = (email or "").strip().lower()
    now = datetime.now(timezone.utc)
    attempts = login_attempts.get(normalized, [])
    attempts = [ts for ts in attempts if ts > now - timedelta(minutes=Config.LOGIN_LOCKOUT_MINUTES)]
    login_attempts[normalized] = attempts
    if len(attempts) >= Config.LOGIN_MAX_ATTEMPTS:
        return True
    return False


def record_login_failure(email):
    """Track a failed login for rate limiting."""
    normalized = (email or "").strip().lower()
    login_attempts[normalized].append(datetime.now(timezone.utc))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("signin", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("signin", next=request.path))
        if session.get("role") != "admin":
            return redirect(url_for("signin"))
        return view(*args, **kwargs)
    return wrapped


def create_app(config_class=Config):
    """Application factory for FREQUENCY."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    app.config.from_object(config_class)
    app.secret_key = app.config["SECRET_KEY"]
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Initialize LLM Client & Database
    llm_client = LLMClient(
        base_url=app.config["OLLAMA_BASE_URL"],
        default_model=app.config["DEFAULT_LLM_MODEL"]
    )
    Database.init_db()

    @app.route("/login", methods=["GET", "POST"])
    def login():
        """Authenticate a user and redirect based on role."""
        if request.method == "GET":
            return render_template("login.html", error=None, email="")

        email = sanitize_text(request.form.get("email", "")).lower()
        password = request.form.get("password", "")
        next_url = request.form.get("next") or request.args.get("next") or ""

        if not email or not password:
            return render_template("login.html", error="Please enter both email and password.", email=email), 400

        if not re.fullmatch(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return render_template("login.html", error="Please enter a valid email address.", email=email), 400

        if rate_limited(email):
            return render_template("login.html", error="Too many attempts. Please wait 15 minutes before trying again.", email=email), 429

        user = Database.authenticate_user(email, password)
        if not user:
            record_login_failure(email)
            return render_template("login.html", error="Invalid email or password.", email=email), 401

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["role"] = user["role"]

        redirect_target = "/admin" if user["role"] == "admin" else "/"
        if next_url and next_url.startswith("/") and next_url not in ("/login", "/signin", "/logout"):
            redirect_target = next_url
        return redirect(redirect_target)

    @app.route("/signin", methods=["GET", "POST"])
    def signin():
        """Alias sign-in page for the app's authentication flow."""
        return login()

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("signin"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        """Create a new user account and redirect to the main app once registered."""
        if request.method == "GET":
            return render_template("signup.html", error=None, name="", email="")

        name = sanitize_text(request.form.get("name", ""))
        email = sanitize_text(request.form.get("email", "")).lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            return render_template("signup.html", error="Name, email, and password are required.", name=name, email=email), 400

        if not re.fullmatch(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return render_template("signup.html", error="Please enter a valid email address.", name=name, email=email), 400

        if len(password) < 8:
            return render_template("signup.html", error="Password must be at least 8 characters long.", name=name, email=email), 400

        try:
            user = Database.create_user(name=name, email=email, password=password, role="user")
        except ValueError as exc:
            return render_template("signup.html", error=str(exc), name=name, email=email), 400

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["role"] = user["role"]
        return redirect(url_for("index"))

    @app.route("/admin")
    @admin_required
    def admin_dashboard():
        """Admin-only dashboard with user and entry analytics."""
        stats = Database.get_dashboard_stats()
        users = Database.list_users()
        entries = Database.list_entries()
        return render_template(
            "admin.html",
            app_name=app.config["APP_NAME"],
            version=app.config["APP_VERSION"],
            stats=stats,
            users=users,
            entries=entries,
            admin_name=session.get("user_name", "Administrator")
        )

    @app.route("/")
    def index():
        """Serve the main FREQUENCY conversational reasoning web interface."""
        user_name = session.get("user_name")
        user_role = session.get("role")
        return render_template(
            "index.html",
            app_name=app.config["APP_NAME"],
            version=app.config["APP_VERSION"],
            default_model=app.config["DEFAULT_LLM_MODEL"],
            user_name=user_name,
            user_role=user_role,
            is_authenticated=bool(session.get("user_id"))
        )

    @app.route("/api/status", methods=["GET"])
    def get_status():
        """Health and status endpoint."""
        available_models = llm_client.get_available_models()
        return jsonify({
            "status": "operational",
            "app_name": app.config["APP_NAME"],
            "version": app.config["APP_VERSION"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning_backend": {
                "provider": "ollama",
                "base_url": app.config["OLLAMA_BASE_URL"],
                "active_model": app.config["DEFAULT_LLM_MODEL"],
                "available_models": available_models,
                "connected": len(available_models) > 0,
                "tuning": {
                    "num_ctx": app.config["OLLAMA_NUM_CTX"],
                    "num_predict": app.config["OLLAMA_NUM_PREDICT"],
                    "num_thread": app.config["OLLAMA_NUM_THREAD"]
                }
            },
            "schema": {
                "version": "1.4.0",
                "endpoint": "/api/schema",
                "prompt_builder": "v1.4-cross-sense-fusion"
            },
            "modalities": {
                "text": "active_phase2",
                "voice": "active_phase3",
                "context": "active_phase4",
                "baseline": "active_phase5",
                "fusion": "active_phase6"
            }
        }), 200

    @app.route("/api/models", methods=["GET"])
    def get_models():
        """Retrieve list of available local models from Ollama."""
        models = llm_client.get_available_models()
        return jsonify({
            "models": models,
            "active_model": app.config["DEFAULT_LLM_MODEL"],
            "count": len(models)
        }), 200

    @app.route("/api/schema", methods=["GET"])
    def get_schema():
        """Expose the JSON schema contract for FrequencyInput evidence packages."""
        return jsonify(FrequencyInput.get_json_schema()), 200

    @app.route("/api/context", methods=["GET", "POST"])
    def handle_context():
        """
        Get or update user's daily lifestyle context (Phase 4).
        """
        user_id = request.args.get("user_id", "default_user")

        if request.method == "POST":
            if not request.is_json:
                return jsonify({"error": "Request payload must be JSON."}), 400
            
            data = request.get_json()
            saved = Database.save_context(
                user_id=user_id,
                sleep_hours=float(data.get("sleep_hours", 7.5)),
                sleep_quality=data.get("sleep_quality", "moderate"),
                workload=data.get("workload", "normal"),
                activity_level=data.get("activity_level", "moderate"),
                mood_score=int(data.get("mood_score", 7)),
                notes=data.get("notes", "")
            )
            return jsonify({
                "status": "success",
                "context": saved,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200

        current_context = Database.get_context(user_id=user_id)
        return jsonify({
            "status": "success",
            "context": current_context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/baseline", methods=["GET"])
    def get_baseline():
        """
        Retrieve personal baseline profile and calculated deviations (Phase 5).
        """
        user_id = request.args.get("user_id", "default_user")
        context_data = Database.get_context(user_id=user_id)
        baseline_with_deviations = BaselineEngine.compute_deviations_and_correlations(
            context_data=context_data
        )
        return jsonify({
            "status": "success",
            "baseline": baseline_with_deviations.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/trends", methods=["GET"])
    def get_trends():
        """
        Retrieve 7-day multi-modal trends for dashboard (Phase 6).
        """
        user_id = request.args.get("user_id", "default_user")
        trends = Database.get_7day_trends(user_id=user_id)
        return jsonify({
            "status": "success",
            "trends": trends,
            "count": len(trends),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/fusion/synthesize", methods=["POST"])
    def synthesize_fusion():
        """
        Execute Cross-Sense Fusion on a FrequencyInput package (Phase 6).
        """
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON."}), 400

        data = request.get_json()
        input_pkg = FrequencyInput.from_dict(data)

        if not input_pkg.text_data and input_pkg.user_prompt:
            input_pkg.text_data = TextAnalyzer.analyze(input_pkg.user_prompt)

        if not input_pkg.contextual_data:
            input_pkg.contextual_data = Database.get_context(user_id="default_user")

        fusion_res = FusionEngine.synthesize(input_pkg)
        return jsonify({
            "status": "success",
            "fusion": fusion_res.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/analyze/text", methods=["POST"])
    def analyze_text():
        """
        Standalone endpoint for local NLP text signal extraction (Phase 2).
        """
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON."}), 400

        data = request.get_json()
        raw_text = data.get("text", "").strip()

        if not raw_text:
            return jsonify({"error": "Text field cannot be empty."}), 400

        text_signals = TextAnalyzer.analyze(raw_text)
        return jsonify({
            "status": "success",
            "signals": text_signals.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/analyze/voice", methods=["POST"])
    def analyze_voice():
        """
        Endpoint for Voice Modality (Phase 3).
        """
        if "audio" not in request.files and "file" not in request.files:
            return jsonify({"error": "Audio file required under 'audio' or 'file' key."}), 400

        audio_file = request.files.get("audio") or request.files.get("file")
        filename = audio_file.filename or "recording.wav"
        audio_bytes = audio_file.read()

        if len(audio_bytes) == 0:
            return jsonify({"error": "Uploaded audio file is empty."}), 400

        client_transcript = request.form.get("transcript", None)

        try:
            voice_signals, transcript = VoiceAnalyzer.analyze_audio_data(
                audio_bytes=audio_bytes,
                filename=filename,
                client_transcript=client_transcript
            )

            text_signals = TextAnalyzer.analyze(transcript) if transcript else None

            return jsonify({
                "status": "success",
                "transcript": transcript,
                "voice_signals": voice_signals.to_dict(),
                "text_signals": text_signals.to_dict() if text_signals else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        except Exception as e:
            logger.error("Error analyzing voice audio: %s", str(e), exc_info=True)
            return jsonify({"error": f"Failed to analyze audio: {str(e)}"}), 500

    @app.route("/api/prompt/preview", methods=["POST"])
    def preview_prompt():
        """
        Compile and preview how a FrequencyInput package translates into an LLM prompt.
        """
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON."}), 400

        data = request.get_json()
        input_package = FrequencyInput.from_dict(data)

        if not input_package.text_data and input_package.user_prompt:
            input_package.text_data = TextAnalyzer.analyze(input_package.user_prompt)

        if not input_package.fusion_data:
            input_package.fusion_data = FusionEngine.synthesize(input_package)

        is_valid, error_msg = input_package.validate()
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        preview_data = PromptBuilder.preview(input_package)
        return jsonify(preview_data), 200

    @app.route("/api/analyze/package", methods=["POST"])
    def parse_package():
        """
        Validate and assemble a structured FrequencyInput package.
        """
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON."}), 400

        data = request.get_json()
        input_package = FrequencyInput.from_dict(data)

        if not input_package.text_data and input_package.user_prompt:
            input_package.text_data = TextAnalyzer.analyze(input_package.user_prompt)

        if not input_package.fusion_data:
            input_package.fusion_data = FusionEngine.synthesize(input_package)

        is_valid, error_msg = input_package.validate()
        if not is_valid:
            return jsonify({
                "status": "error",
                "error": error_msg
            }), 400

        return jsonify({
            "status": "valid",
            "package": input_package.to_dict(),
            "active_modalities": input_package.get_active_modalities(),
            "compiled_prompt": PromptBuilder.build_evidence_prompt(input_package),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/chat/stream", methods=["POST"])
    def chat_stream():
        """
        Real-time streaming conversational reasoning endpoint with full multimodal cross-sense synthesis.
        """
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON."}), 400

        data = request.get_json()
        model_override = data.get("model", None)

        if "user_prompt" in data:
            input_pkg = FrequencyInput.from_dict(data)
        else:
            raw_msg = data.get("message", "").strip()
            input_pkg = FrequencyInput(user_prompt=raw_msg)

        # 1. Text signal analysis
        if not input_pkg.text_data and input_pkg.user_prompt:
            input_pkg.text_data = TextAnalyzer.analyze(input_pkg.user_prompt)

        # 2. Context Auto-Population (Phase 4)
        if not input_pkg.contextual_data:
            input_pkg.contextual_data = Database.get_context(user_id="default_user")

        # 3. Baseline & Cross-Modal Deviations (Phase 5)
        raw_context = input_pkg.contextual_data.to_dict() if isinstance(input_pkg.contextual_data, ContextData) else input_pkg.contextual_data
        raw_text = input_pkg.text_data.to_dict() if isinstance(input_pkg.text_data, TextSignalData) else input_pkg.text_data
        raw_voice = input_pkg.voice_data.to_dict() if isinstance(input_pkg.voice_data, VoiceSignalData) else input_pkg.voice_data

        input_pkg.baseline_data = BaselineEngine.compute_deviations_and_correlations(
            context_data=raw_context,
            text_data=raw_text,
            voice_data=raw_voice
        )

        # 4. Cross-Sense Fusion Matrix (Phase 6)
        if not input_pkg.fusion_data:
            input_pkg.fusion_data = FusionEngine.synthesize(input_pkg)

        is_valid, err = input_pkg.validate()
        if not is_valid:
            return jsonify({"error": err}), 400

        # Log session to database
        try:
            modality_label = "voice" if input_pkg.voice_data else "text"
            session_id = input_pkg.metadata.get("session_id", "session_stream")
            Database.log_session(
                session_id=session_id,
                prompt=input_pkg.user_prompt,
                modality=modality_label,
                text_signals=raw_text,
                voice_signals=raw_voice
            )
        except Exception as e:
            logger.warning("Failed to log session history: %s", str(e))

        compiled_evidence = PromptBuilder.build_evidence_prompt(input_pkg)
        system_prompt = PromptBuilder.build_system_prompt()

        logger.info("Streaming reasoning request for prompt: '%s' (model: %s)", input_pkg.user_prompt[:50], model_override or app.config["DEFAULT_LLM_MODEL"])

        def generate_sse():
            for chunk_data in llm_client.stream_generate(
                prompt=compiled_evidence,
                model=model_override,
                system_prompt=system_prompt
            ):
                yield f"data: {json.dumps(chunk_data)}\n\n"

        return Response(
            stream_with_context(generate_sse()),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat():
        """Synchronous reasoning endpoint fallback."""
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON."}), 400

        data = request.get_json()
        model_override = data.get("model", None)

        if "user_prompt" in data:
            input_pkg = FrequencyInput.from_dict(data)
        else:
            raw_msg = data.get("message", "").strip()
            input_pkg = FrequencyInput(user_prompt=raw_msg)

        if not input_pkg.text_data and input_pkg.user_prompt:
            input_pkg.text_data = TextAnalyzer.analyze(input_pkg.user_prompt)

        if not input_pkg.contextual_data:
            input_pkg.contextual_data = Database.get_context(user_id="default_user")

        raw_context = input_pkg.contextual_data.to_dict() if isinstance(input_pkg.contextual_data, ContextData) else input_pkg.contextual_data
        raw_text = input_pkg.text_data.to_dict() if isinstance(input_pkg.text_data, TextSignalData) else input_pkg.text_data
        raw_voice = input_pkg.voice_data.to_dict() if isinstance(input_pkg.voice_data, VoiceSignalData) else input_pkg.voice_data

        input_pkg.baseline_data = BaselineEngine.compute_deviations_and_correlations(
            context_data=raw_context,
            text_data=raw_text,
            voice_data=raw_voice
        )

        if not input_pkg.fusion_data:
            input_pkg.fusion_data = FusionEngine.synthesize(input_pkg)

        is_valid, err = input_pkg.validate()
        if not is_valid:
            return jsonify({"error": err}), 400

        compiled_evidence = PromptBuilder.build_evidence_prompt(input_pkg)
        system_prompt = PromptBuilder.build_system_prompt()

        result = llm_client.generate_response(
            prompt=compiled_evidence,
            model=model_override,
            system_prompt=system_prompt
        )

        if result.get("error"):
            return jsonify({
                "status": "error",
                "error": result["error"],
                "model": result["model"],
                "duration_ms": result["duration_ms"]
            }), 502

        return jsonify({
            "status": "success",
            "reply": result["reply"],
            "model": result["model"],
            "duration_ms": result["duration_ms"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    return app

app = create_app()

if __name__ == "__main__":
    logger.info("Starting FREQUENCY server on %s:%s", Config.HOST, Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=False, use_reloader=False)
