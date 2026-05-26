"""Thin HTTP client for the local Ollama sidecar.

We use the Ollama REST API directly via :mod:`httpx` instead of pulling
in the ``ollama`` Python package — the surface we need is one POST to
``/api/generate`` and a health probe. Keeping the dependency surface
small means the API container starts without `ollama` installed and
the LLM features simply don't appear.

Graceful degradation
--------------------
Callers should treat any :class:`OllamaUnavailable` as "feature not
ready" and surface the absence (e.g. hide the LLM preview card on the
match page; return a 503 from the explainer endpoint). The class
hierarchy below makes that easy:

* :class:`OllamaUnavailable` — the sidecar is unreachable or hasn't
  loaded the model yet. Retry later.
* :class:`OllamaGenerationError` — the sidecar responded but the
  generation itself failed (model loading OOM, malformed prompt, etc.).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: float = 60.0
DEFAULT_MAX_TOKENS: int = 200
DEFAULT_MODEL: str = "mistral:7b-instruct-q4_K_M"

OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"


class OllamaUnavailable(RuntimeError):
    """The sidecar can't be reached (network / not started / model loading)."""


class OllamaGenerationError(RuntimeError):
    """Sidecar responded but the generation step failed."""


@dataclass(frozen=True)
class OllamaConfig:
    """Resolved Ollama connection config from env + caller overrides."""

    base_url: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls) -> OllamaConfig | None:
        """Build a config from env vars; ``None`` when ``OLLAMA_BASE_URL`` is unset.

        Returning ``None`` is the explicit "feature off" signal — the API
        container ships without LLM previews in environments where the
        sidecar isn't deployed.
        """
        base_url = os.environ.get(OLLAMA_BASE_URL_ENV, "").strip()
        if not base_url:
            return None
        model = os.environ.get(OLLAMA_MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return cls(base_url=base_url.rstrip("/"), model=model)


def is_available(cfg: OllamaConfig | None = None, *, timeout_s: float = 5.0) -> bool:
    """Cheap health probe. Returns ``False`` on any connection / parse error."""
    cfg = cfg or OllamaConfig.from_env()
    if cfg is None:
        return False
    try:
        resp = httpx.get(f"{cfg.base_url}/api/tags", timeout=timeout_s)
        resp.raise_for_status()
        # Body is `{"models": [...]}`; presence-only check is enough.
        return "models" in resp.json()
    except (httpx.HTTPError, ValueError):
        return False


def generate(
    prompt: str,
    *,
    cfg: OllamaConfig | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
) -> str:
    """POST to ``/api/generate`` and return the generated string.

    Raises
    ------
    OllamaUnavailable
        When the sidecar can't be reached or doesn't respond in time.
    OllamaGenerationError
        When the sidecar responds with a non-2xx, or a 2xx whose body
        doesn't carry the expected ``response`` field.
    """
    cfg = cfg or OllamaConfig.from_env()
    if cfg is None:
        raise OllamaUnavailable(
            f"{OLLAMA_BASE_URL_ENV} is unset; LLM features are disabled"
        )
    payload: dict[str, Any] = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = httpx.post(
            f"{cfg.base_url}/api/generate",
            json=payload,
            timeout=cfg.timeout_s,
        )
    except httpx.HTTPError as exc:
        logger.warning("ollama unreachable at %s: %s", cfg.base_url, exc)
        raise OllamaUnavailable(str(exc)) from exc
    if resp.status_code >= 500:
        # 5xx is typically "model still loading" or "OOM" — retryable.
        raise OllamaUnavailable(
            f"ollama returned {resp.status_code}; model may still be loading"
        )
    if resp.status_code >= 400:
        raise OllamaGenerationError(
            f"ollama rejected the generation request: {resp.status_code} {resp.text[:200]}"
        )
    try:
        body = resp.json()
    except ValueError as exc:
        raise OllamaGenerationError(f"ollama response was not JSON: {exc}") from exc
    text = body.get("response")
    if not isinstance(text, str):
        raise OllamaGenerationError(
            f"ollama response missing 'response' field; got keys {list(body.keys())!r}"
        )
    return text.strip()


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_MODEL_ENV",
    "OllamaConfig",
    "OllamaGenerationError",
    "OllamaUnavailable",
    "generate",
    "is_available",
]
