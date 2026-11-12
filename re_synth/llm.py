"""LLM access via D:/NLP/llm_providers (OpenRouter, or the Copilot CLI for `copilot-<model>`) with an on-disk cache.

Env: RE_ENV_FILE=<path to .env containing OPENROUTER_API_KEY> (optional; plain env vars also work).
     PYTHONPATH must include the llm_providers repo (or `pip install -e <llm_providers>`).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading

from dotenv import load_dotenv

log = logging.getLogger(__name__)
_env_loaded = False
_lock = threading.Lock()

MODELS = {
    "gen": "google/gemini-3.7-flash",
    "weak": "meta-llama/llama-3.3-70b-instruct",
    "strong": "anthropic/claude-sonnet-5.5",
    "strong2": "openai/gpt-5.6-terra",
    "cheap": "deepseek/deepseek-v4.1-flash",
}


def _ensure_env():
    global _env_loaded
    with _lock:
        if not _env_loaded:
            p = os.environ.get("RE_ENV_FILE")
            if p and os.path.exists(p):
                load_dotenv(p, override=False)
            _env_loaded = True


_cache = None


def _get_cache():
    global _cache
    if _cache is None:
        from llm_providers.llm_cache import LLMCache
        _cache = LLMCache(os.environ.get("RE_LLM_CACHE", "cache/llm"), logger=logging.getLogger("llm_cache"))
    return _cache


def chat(messages: list[dict], model: str, temperature: float = 0.0, max_tokens: int = 4000,
         json_mode: bool = False, salt: str | int | None = None, use_cache: bool = True,
         reasoning_effort: str | None = None) -> str:
    """`salt` differentiates cache keys for repeated sampling (e.g. different seeds)."""
    _ensure_env()
    model = MODELS.get(model, model)
    rf = {"type": "json_object"} if json_mode else None

    if model.startswith("copilot-"):  # GitHub Copilot CLI (no temperature / json_mode / max_tokens flags)
        def producer():
            from llm_providers.copilot_cli import call_copilot
            return call_copilot(messages, model[len("copilot-"):], reasoning_effort=reasoning_effort)
    else:
        def producer():
            from llm_providers.openrouter import call_openrouter
            return call_openrouter(messages, model, max_tokens=max_tokens, temperature=temperature,
                                   response_format=rf, reasoning_effort=reasoning_effort)

    if not use_cache:
        return producer()
    params = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode, "salt": salt,
              "reasoning": reasoning_effort}
    return _get_cache().get(model, messages, params, None, producer)


def parse_json_obj(s: str) -> dict | None:
    """Lenient: strips code fences / <think> blocks, takes the outermost {...}. Never raises."""
    if not isinstance(s, str):
        return None
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            o = json.loads(s[i:j + 1])
            return o if isinstance(o, dict) else None
        except Exception:
            return None
    return None


def usage_totals():
    try:
        from llm_providers import usage
        return usage.totals()
    except Exception:
        return None
