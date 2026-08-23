import json
import logging
import time
import requests
from config import Config

logger = logging.getLogger("frequency.llm_client")

class LLMClient:
    """Decoupled client for interacting with local LLM reasoning backends (e.g. Ollama)."""

    def __init__(self, base_url=None, default_model=None):
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = default_model or Config.DEFAULT_LLM_MODEL

    def get_default_options(self, custom_options=None):
        """Construct optimized runtime options for Ollama."""
        opts = {
            "num_ctx": Config.OLLAMA_NUM_CTX,
            "num_predict": Config.OLLAMA_NUM_PREDICT,
            "num_thread": Config.OLLAMA_NUM_THREAD,
            "temperature": Config.OLLAMA_TEMPERATURE
        }
        if custom_options:
            opts.update(custom_options)
        return opts

    def get_available_models(self):
        """Fetch list of available models from Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", [])]
                return models
            logger.warning("Failed to fetch models: %s", response.status_code)
            return []
        except requests.exceptions.RequestException as e:
            logger.error("Error connecting to Ollama tags endpoint: %s", e)
            return []

    def stream_generate(self, prompt, model=None, system_prompt=None, options=None):
        """
        Stream response tokens in real-time from Ollama.
        Yields dicts with {'chunk': str, 'done': bool, 'duration_ms': int, 'model': str}
        """
        target_model = model or self.default_model
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": Config.OLLAMA_KEEP_ALIVE,
            "options": self.get_default_options(options)
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        in_thinking_block = False

        try:
            logger.info("Streaming request from Ollama (model: %s)...", target_model)
            with requests.post(url, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    yield {
                        "chunk": None,
                        "done": True,
                        "error": f"Ollama HTTP error {resp.status_code}",
                        "model": target_model,
                        "duration_ms": int((time.time() - start_time) * 1000)
                    }
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk_obj = json.loads(line.decode("utf-8"))
                        text_chunk = chunk_obj.get("response", "")
                        done = chunk_obj.get("done", False)

                        # Handle reasoning/thinking tags gracefully if emitted
                        if "<think>" in text_chunk:
                            in_thinking_block = True
                            text_chunk = text_chunk.replace("<think>", "")
                        if "</think>" in text_chunk:
                            in_thinking_block = False
                            text_chunk = text_chunk.replace("</think>", "")

                        # Yield non-empty chunks
                        if text_chunk and not in_thinking_block:
                            yield {
                                "chunk": text_chunk,
                                "done": False,
                                "model": target_model
                            }

                        if done:
                            duration_ms = int((time.time() - start_time) * 1000)
                            yield {
                                "chunk": "",
                                "done": True,
                                "model": target_model,
                                "duration_ms": duration_ms,
                                "eval_count": chunk_obj.get("eval_count", 0)
                            }
                            return
                    except Exception as parse_err:
                        logger.warning("Error parsing stream line: %s", parse_err)

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Streaming error connecting to Ollama: %s", e)
            yield {
                "chunk": None,
                "done": True,
                "error": str(e),
                "model": target_model,
                "duration_ms": duration_ms
            }

    def generate_response(self, prompt, model=None, system_prompt=None, options=None, timeout=120):
        """
        Synchronous fallback for non-streaming callers.
        """
        target_model = model or self.default_model
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": Config.OLLAMA_KEEP_ALIVE,
            "options": self.get_default_options(options)
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        try:
            logger.info("Sending synchronous request to Ollama (model: %s)...", target_model)
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
            duration_ms = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                error_msg = f"Ollama returned HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {
                    "reply": None,
                    "model": target_model,
                    "duration_ms": duration_ms,
                    "error": error_msg
                }

            data = response.json()
            raw_reply = data.get("response", "")
            return {
                "reply": raw_reply,
                "model": target_model,
                "duration_ms": duration_ms,
                "error": None
            }
        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return {
                "reply": None,
                "model": target_model,
                "duration_ms": duration_ms,
                "error": str(e)
            }
