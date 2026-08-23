import re
from typing import Dict, Any, List, Optional
from schema import TextSignalData

class TextAnalyzer:
    """
    Local, deterministic NLP Signal Extractor for FREQUENCY.
    Extracts sentiment valence, emotional affect markers, themes, and cognitive load metrics.
    """

    # Sentiment lexicon weights (with support for stems/prefixes)
    POSITIVE_LEXICON = {
        "great": 0.8, "good": 0.5, "excellent": 0.9, "happy": 0.7, "happier": 0.8, "happiest": 0.9,
        "energiz": 0.8, "energy": 0.6, "excit": 0.8, "focus": 0.6, "focused": 0.7, "focusing": 0.7,
        "productiv": 0.8, "calm": 0.6, "peace": 0.7, "peaceful": 0.7, "rest": 0.5, "rested": 0.8,
        "motivat": 0.8, "confiden": 0.7, "inspir": 0.8, "refresh": 0.8, "clear": 0.5, "clarity": 0.7,
        "relax": 0.6, "joy": 0.8, "love": 0.8, "wonder": 0.8, "wonderful": 0.9, "sharp": 0.6,
        "optimis": 0.7, "accomplish": 0.7, "vibran": 0.8, "thriv": 0.9, "well": 0.5, "better": 0.6,
        "best": 0.8, "super": 0.7, "awesome": 0.8, "relief": 0.7, "relieved": 0.7, "recovering": 0.7
    }

    NEGATIVE_LEXICON = {
        "exhaust": -0.9, "exhausted": -0.9, "exhausting": -0.9, "exhaustion": -0.9,
        "tire": -0.6, "tired": -0.6, "tiring": -0.7, "fatigue": -0.8, "fatigued": -0.8,
        "drain": -0.8, "drained": -0.8, "draining": -0.8, "stress": -0.8, "stressed": -0.8,
        "stressful": -0.8, "anxi": -0.7, "anxious": -0.7, "anxiety": -0.8,
        "overwhelm": -0.9, "overwhelmed": -0.9, "overwhelming": -0.9,
        "sad": -0.7, "sadness": -0.8, "depress": -0.9, "depressed": -0.9, "depressing": -0.9,
        "frustrat": -0.7, "frustrated": -0.7, "frustrating": -0.8, "frustration": -0.8,
        "angr": -0.7, "angry": -0.7, "fog": -0.6, "foggy": -0.7, "sluggish": -0.7,
        "unfocus": -0.6, "unfocused": -0.7, "scatter": -0.6, "scattered": -0.6,
        "burnt": -0.8, "burnout": -0.9, "restless": -0.5, "hopeless": -0.9,
        "irritat": -0.6, "irritated": -0.6, "irritating": -0.7,
        "distract": -0.5, "distracted": -0.6, "distraction": -0.6,
        "struggl": -0.7, "struggled": -0.7, "struggling": -0.7, "struggle": -0.7,
        "awful": -0.8, "terrible": -0.8, "miserab": -0.9, "miserable": -0.9,
        "drag": -0.6, "dragged": -0.7, "dragging": -0.7, "bad": -0.6, "worse": -0.7, "worst": -0.9,
        "pain": -0.7, "hurting": -0.6, "heavy": -0.5, "stuck": -0.6, "low": -0.5
    }

    # Emotion categories and indicator lexicons
    EMOTION_LEXICON = {
        "exhaustion_fatigue": [
            "exhaust", "tire", "fatigue", "drain", "sleepy", "lethargic", "sluggish",
            "weary", "spent", "wiped", "burnt out", "no energy", "low energy", "worn out",
            "dragged", "dragging"
        ],
        "stress_anxiety": [
            "stress", "anxi", "overwhelm", "nervous", "pressure", "panic", "tense",
            "deadline", "chaotic", "rushed", "hectic", "frantic", "worry", "worried"
        ],
        "joy_optimism": [
            "happy", "excit", "great", "energiz", "inspir", "confiden", "joy",
            "cheerful", "thrilled", "glad", "optimis", "accomplish", "proud", "awesome"
        ],
        "calm_contentment": [
            "calm", "relax", "peace", "balanced", "centered", "serene", "steady",
            "settled", "content", "composed", "rested", "refreshed", "relief"
        ],
        "frustration_irritation": [
            "frustrat", "annoy", "irritat", "angry", "stuck", "blocked", "upset",
            "mad", "impatient", "bothered"
        ],
        "low_mood_sadness": [
            "sad", "down", "low", "gloomy", "unhappy", "depress", "blue", "hopeless",
            "discouraged", "lonely", "miserab"
        ]
    }

    # Thematic keywords
    THEME_LEXICON = {
        "workload_career": [
            "work", "worked", "working", "workload", "job", "project", "deadline", "meeting", "boss", "client", "office",
            "task", "deliverable", "code", "report", "presentation", "career", "shift", "busy", "labor"
        ],
        "sleep_recovery": [
            "sleep", "slept", "sleepy", "insomnia", "nap", "woke", "bed", "rest", "night", "tired",
            "dream", "exhaustion", "awake", "sleeping", "recovery"
        ],
        "focus_productivity": [
            "focus", "focused", "focusing", "concentrate", "distraction", "distracted", "productive", "flow", "unfocused",
            "scattered", "deep work", "brain fog", "clarity", "studying", "thinking", "sluggish"
        ],
        "physical_energy": [
            "workout", "exercise", "run", "gym", "body", "pain", "headache", "diet",
            "meal", "caffeine", "coffee", "active", "stamina", "walk"
        ],
        "social_relational": [
            "friend", "family", "partner", "people", "team", "colleague", "talk", "social",
            "argument", "support", "lonely", "dinner", "party"
        ]
    }

    CERTAINTY_WORDS = {"definitely", "clearly", "certainly", "absolutely", "always", "know", "confident"}
    HESITATION_WORDS = {"maybe", "perhaps", "guess", "confused", "unclear", "doubt", "hardly", "wondering", "unsure"}

    @classmethod
    def _match_lexicon(cls, word: str, lexicon: Dict[str, float]) -> Optional[float]:
        """Check direct match or stem prefix match against lexicon."""
        if word in lexicon:
            return lexicon[word]
        for key, val in lexicon.items():
            if len(key) >= 4 and word.startswith(key):
                return val
        return None

    @classmethod
    def extract_sentiment(cls, text: str) -> Dict[str, Any]:
        """Calculate sentiment valence score (-1.0 to 1.0) and polarity category."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {"valence_score": 0.0, "polarity": "neutral", "intensity": 0.0}

        score = 0.0
        matches = 0

        for i, word in enumerate(words):
            multiplier = 1.0
            # Check for negation (e.g. "not happy", "couldn't focus", "barely slept")
            if i > 0 and (words[i - 1] in {"not", "never", "no", "hardly", "barely", "scarcely", "couldnt", "cannot"} or words[i - 1].endswith("n't")):
                multiplier = -0.7

            pos_val = cls._match_lexicon(word, cls.POSITIVE_LEXICON)
            neg_val = cls._match_lexicon(word, cls.NEGATIVE_LEXICON)

            if neg_val is not None:
                score += neg_val * multiplier
                matches += 1
            elif pos_val is not None:
                score += pos_val * multiplier
                matches += 1

        if matches == 0:
            valence = 0.0
        else:
            valence = max(-1.0, min(1.0, score / max(1, matches)))

        if valence >= 0.2:
            polarity = "positive"
        elif valence <= -0.2:
            polarity = "negative"
        else:
            polarity = "neutral"

        return {
            "valence_score": round(valence, 3),
            "polarity": polarity,
            "matched_terms": matches,
            "intensity": round(abs(valence), 3)
        }

    @classmethod
    def extract_emotions(cls, text: str) -> Dict[str, float]:
        """Compute relative presence (0.0 to 1.0) for primary emotional affect categories."""
        lower_text = text.lower()
        emotion_scores = {}

        for emotion_name, keywords in cls.EMOTION_LEXICON.items():
            count = 0
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw), lower_text):
                    count += 1
            if count > 0:
                emotion_scores[emotion_name] = round(min(1.0, 0.4 + (count * 0.3)), 2)

        return emotion_scores

    @classmethod
    def extract_themes(cls, text: str) -> List[str]:
        """Identify key topics/themes present in the text."""
        lower_text = text.lower()
        detected = []

        for theme_name, keywords in cls.THEME_LEXICON.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw), lower_text):
                    detected.append(theme_name)
                    break

        return detected

    @classmethod
    def extract_linguistic_metrics(cls, text: str) -> Dict[str, Any]:
        """Compute linguistic characteristics and cognitive load markers."""
        words = re.findall(r'\b\w+\b', text)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_sentence_len = round(word_count / sentence_count, 1)

        lower_words = set(w.lower() for w in words)
        certainty_count = sum(1 for w in lower_words if w in cls.CERTAINTY_WORDS)
        hesitation_count = sum(1 for w in lower_words if w in cls.HESITATION_WORDS)

        exclamations = text.count('!')
        questions = text.count('?')

        if avg_sentence_len > 18 or hesitation_count >= 2:
            cognitive_load = "elevated"
        elif avg_sentence_len < 6 and word_count > 10:
            cognitive_load = "fragmented"
        else:
            cognitive_load = "moderate"

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_len,
            "certainty_markers": certainty_count,
            "hesitation_markers": hesitation_count,
            "cognitive_load": cognitive_load,
            "exclamation_count": exclamations,
            "question_count": questions
        }

    @classmethod
    def analyze(cls, text: str) -> TextSignalData:
        """
        Execute full text analysis pipeline and return structured TextSignalData.
        """
        if not text or not text.strip():
            return TextSignalData(raw_text="")

        cleaned = text.strip()
        sentiment_res = cls.extract_sentiment(cleaned)
        emotions = cls.extract_emotions(cleaned)
        themes = cls.extract_themes(cleaned)
        metrics = cls.extract_linguistic_metrics(cleaned)

        cog_markers = []
        if metrics["hesitation_markers"] > 0:
            cog_markers.append(f"Hesitation indicators ({metrics['hesitation_markers']})")
        if metrics["certainty_markers"] > 0:
            cog_markers.append(f"Certainty indicators ({metrics['certainty_markers']})")
        if metrics["exclamation_count"] > 1:
            cog_markers.append("High punctuation intensity")

        return TextSignalData(
            raw_text=cleaned,
            sentiment=sentiment_res["polarity"],
            emotional_valence=sentiment_res["valence_score"],
            emotional_signals=emotions,
            detected_themes=themes,
            cognitive_load=metrics["cognitive_load"],
            cognitive_load_markers=cog_markers,
            word_count=metrics["word_count"],
            confidence=0.9 if metrics["word_count"] >= 4 else 0.5
        )
