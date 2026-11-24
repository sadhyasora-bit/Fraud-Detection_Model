"""Gemini Interactions API client for CreditGuard AI.

Keeps generate_response(prompt) backward-compatible while enforcing a
bounded request timeout and disabling SDK-level automatic retries.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Project root: app/
APP_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(APP_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

try:
    DEFAULT_TIMEOUT_SECONDS = max(
        5, int(os.getenv("GEMINI_TIMEOUT_SECONDS", "40"))
    )
except (TypeError, ValueError):
    DEFAULT_TIMEOUT_SECONDS = 40


class GeminiConfigurationError(RuntimeError):
    """Missing Gemini credentials or configuration."""


class GeminiEmptyResponseError(RuntimeError):
    """Gemini returned no usable text."""


def get_gemini_client(timeout: Optional[int] = None):
    """Create a Gemini client with bounded HTTP timeout and no SDK retries."""
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY tidak ditemukan. Periksa file .env di folder app."
        )

    timeout_seconds = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError("Timeout harus berupa bilangan bulat dalam detik.")
    if timeout_seconds < 5:
        raise ValueError("Timeout minimal adalah 5 detik.")

    # Python Gen AI SDK HttpOptions.timeout is configured in milliseconds.
    # attempts=1 means one HTTP attempt (no automatic retry/backoff).
    http_options = types.HttpOptions(
        timeout=timeout_seconds * 1000,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key=api_key, http_options=http_options)


def get_gemini_config() -> dict:
    """Return safe configuration details; never returns the API key."""
    return {
        "provider": "Google Gemini",
        "api": "Interactions API",
        "model": GEMINI_MODEL,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "sdk_retry_attempts": 1,
        "api_key_configured": bool(
            os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        ),
    }


def generate_response(prompt: str, timeout: Optional[int] = None) -> str:
    """Generate one response through Gemini Interactions API.

    The per-call timeout and HTTP timeout are both bounded. SDK retries are
    disabled so a transient failure does not silently multiply wait time.
    Exceptions are propagated to the caller for fallback handling.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt tidak boleh kosong.")

    request_timeout = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
    if isinstance(request_timeout, bool) or not isinstance(request_timeout, int):
        raise ValueError("Timeout harus berupa bilangan bulat dalam detik.")
    if request_timeout < 5:
        raise ValueError("Timeout minimal adalah 5 detik.")

    started_at = time.perf_counter()
    client = None
    outcome = "error"

    try:
        client = get_gemini_client(timeout=request_timeout)
        interaction = client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt.strip(),
            timeout=request_timeout,
            automatic_continuation=False,
        )

        response_text = getattr(interaction, "output_text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise GeminiEmptyResponseError(
                "Gemini tidak menghasilkan respons teks."
            )

        outcome = "success"
        return response_text.strip()

    finally:
        elapsed = time.perf_counter() - started_at
        print(
            f"[CreditGuard Gemini] model={GEMINI_MODEL} "
            f"status={outcome} elapsed={elapsed:.2f}s "
            f"timeout={request_timeout}s retries=0",
            flush=True,
        )
        if client is not None:
            try:
                client.close()
            except Exception:
                # Closing a client must not hide the original request result/error.
                pass
