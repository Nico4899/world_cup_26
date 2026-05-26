"""Tests for the local-Ollama HTTP client. No live network calls."""

from __future__ import annotations

import httpx
import pytest

from wc2026.llm import ollama_client as oc


# --- OllamaConfig.from_env ---------------------------------------------------


def test_from_env_returns_none_when_base_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(oc.OLLAMA_BASE_URL_ENV, raising=False)
    assert oc.OllamaConfig.from_env() is None


def test_from_env_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oc.OLLAMA_BASE_URL_ENV, "http://ollama:11434/")
    cfg = oc.OllamaConfig.from_env()
    assert cfg is not None
    assert cfg.base_url == "http://ollama:11434"


def test_from_env_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oc.OLLAMA_BASE_URL_ENV, "http://ollama:11434")
    monkeypatch.delenv(oc.OLLAMA_MODEL_ENV, raising=False)
    cfg = oc.OllamaConfig.from_env()
    assert cfg is not None
    assert cfg.model == oc.DEFAULT_MODEL


def test_from_env_uses_override_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oc.OLLAMA_BASE_URL_ENV, "http://ollama:11434")
    monkeypatch.setenv(oc.OLLAMA_MODEL_ENV, "llama3.1:8b-instruct-q4_K_M")
    cfg = oc.OllamaConfig.from_env()
    assert cfg is not None
    assert cfg.model == "llama3.1:8b-instruct-q4_K_M"


# --- is_available ------------------------------------------------------------


def test_is_available_false_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(oc.OLLAMA_BASE_URL_ENV, raising=False)
    assert oc.is_available() is False


def test_is_available_returns_true_when_tags_endpoint_responds_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[object]]:
            return {"models": []}

    def fake_get(url: str, *, timeout: float) -> FakeResp:
        assert url.endswith("/api/tags")
        return FakeResp()

    monkeypatch.setattr(httpx, "get", fake_get)
    cfg = oc.OllamaConfig(base_url="http://ollama:11434", model="mistral:7b")
    assert oc.is_available(cfg) is True


def test_is_available_returns_false_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kw: object) -> object:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", boom)
    cfg = oc.OllamaConfig(base_url="http://ollama:11434", model="mistral:7b")
    assert oc.is_available(cfg) is False


# --- generate ---------------------------------------------------------------


def test_generate_raises_unavailable_when_env_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(oc.OLLAMA_BASE_URL_ENV, raising=False)
    with pytest.raises(oc.OllamaUnavailable, match="is unset"):
        oc.generate("hi")


def test_generate_returns_response_field(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResp:
        status_code = 200
        text = ""

        def json(self) -> dict[str, str]:
            return {"response": "  Argentina is favored.  "}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResp:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    cfg = oc.OllamaConfig(base_url="http://ollama:11434", model="mistral:7b")
    out = oc.generate("Write a preview.", cfg=cfg, max_tokens=120)
    assert out == "Argentina is favored."
    assert captured["url"] == "http://ollama:11434/api/generate"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "mistral:7b"
    assert payload["options"]["num_predict"] == 120
    assert payload["stream"] is False


def test_generate_raises_unavailable_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kw: object) -> object:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", boom)
    cfg = oc.OllamaConfig(base_url="http://ollama:11434", model="mistral:7b")
    with pytest.raises(oc.OllamaUnavailable):
        oc.generate("anything", cfg=cfg)


def test_generate_raises_unavailable_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """5xx is treated as a transient retryable failure (model loading)."""

    class FakeResp:
        status_code = 503
        text = "model loading"

        def json(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(httpx, "post", lambda *_a, **_kw: FakeResp())
    cfg = oc.OllamaConfig(base_url="http://x", model="m")
    with pytest.raises(oc.OllamaUnavailable):
        oc.generate("p", cfg=cfg)


def test_generate_raises_generation_error_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        status_code = 400
        text = "bad model name"

    monkeypatch.setattr(httpx, "post", lambda *_a, **_kw: FakeResp())
    cfg = oc.OllamaConfig(base_url="http://x", model="missing:tag")
    with pytest.raises(oc.OllamaGenerationError, match="400"):
        oc.generate("p", cfg=cfg)


def test_generate_raises_generation_error_on_missing_response_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResp:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            return {"unexpected": "shape"}

    monkeypatch.setattr(httpx, "post", lambda *_a, **_kw: FakeResp())
    cfg = oc.OllamaConfig(base_url="http://x", model="m")
    with pytest.raises(oc.OllamaGenerationError, match="missing 'response'"):
        oc.generate("p", cfg=cfg)
