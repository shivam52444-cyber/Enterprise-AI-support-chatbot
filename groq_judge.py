"""
groq_judge.py - lets DeepEval use a Groq model as its "judge".

DeepEval's LLM-based metrics (Faithfulness, Answer Relevancy, ...) normally call OpenAI.
We only have Groq, so this class plugs Groq in. It follows DeepEval's custom-LLM rules:
  - generate(prompt, schema): return an object of the `schema` class (JSON from the model)
  - generate(prompt): return plain text
  - a_generate: the async twin, get_model_name, load_model

Tip: the judge should ideally NOT be the same model that writes the chatbot's answers
(a model tends to rate its own style higher). Set JUDGE_MODEL in .env to a different one.
Run `python list_groq_models.py` to see what your key can use.
"""

import asyncio
import json
import os
import re
import threading
import time

from deepeval.models import DeepEvalBaseLLM

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
MAX_TRIES = 5
BACKOFF_SECONDS = 5          # waits 5, 10, 20, 40 seconds after a rate-limit error


def extract_json(text):
    """Pull the JSON object out of a model reply (it may add ```json fences or extra words)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def _is_rate_limit(error):
    text = str(error).lower()
    return getattr(error, "status_code", None) == 429 or "429" in text or "rate limit" in text


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model=None, api_key=None, min_interval=None, client=None):
        self.model_name = (model or os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL")
                           or DEFAULT_JUDGE_MODEL)
        if client is None:
            from groq import Groq
            client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.client = client
        # Space the calls out a little so the free tier's rate limit is not hit constantly.
        self.min_interval = float(os.getenv("JUDGE_MIN_INTERVAL", "1.0")
                                  if min_interval is None else min_interval)
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.calls = 0

    def load_model(self):
        return self.client

    def get_model_name(self):
        return f"Groq {self.model_name}"

    def _throttle(self):
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _complete(self, prompt):
        for attempt in range(MAX_TRIES):
            self._throttle()
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0)
                self.calls += 1
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == MAX_TRIES - 1 or not _is_rate_limit(e):
                    raise
                time.sleep(min(60, BACKOFF_SECONDS * 2 ** attempt))

    def generate(self, prompt, schema=None):
        if schema is None:
            return self._complete(prompt)
        try:
            schema_text = json.dumps(schema.model_json_schema())
        except AttributeError:                      # pydantic v1
            schema_text = schema.schema_json()
        full = (prompt + "\n\n---\nReturn ONLY one valid JSON object (no markdown fences, no extra "
                "text) that matches this JSON schema:\n" + schema_text)
        last_error = None
        for attempt in range(2):
            reminder = "" if attempt == 0 else "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."
            try:
                return schema(**extract_json(self._complete(full + reminder)))
            except (ValueError, TypeError) as e:     # bad JSON or wrong fields (pydantic errors are ValueErrors)
                last_error = e
        raise ValueError(f"the judge did not return valid JSON: {last_error}")

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)
