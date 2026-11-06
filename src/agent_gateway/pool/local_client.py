from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .._config import DEFAULT_OLLAMA_URL


class OllamaClientError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class OllamaResponse:
    text: str
    metrics: dict[str, Any]


class OllamaClient(Protocol):
    def generate(
        self,
        *,
        model_tag: str,
        prompt: str,
        timeout_seconds: int,
        maximum_output_tokens: int,
    ) -> OllamaResponse:
        ...


class LocalOllamaClient:
    """A small standard-library adapter for Ollama's ``/api/generate``.

    Construction and import do nothing. Network traffic happens only when
    ``generate()`` is called by an explicitly enabled delegation.
    """

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        *,
        model_tag: str,
        prompt: str,
        timeout_seconds: int,
        maximum_output_tokens: int,
    ) -> OllamaResponse:
        started = time.monotonic()
        payload = {
            "model": model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": maximum_output_tokens},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise OllamaClientError("timeout", str(exc)) from exc
        except urllib.error.HTTPError as exc:
            raise OllamaClientError(_http_error_category(exc.code), str(exc)) from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            category = "timeout" if "timed out" in reason.lower() else "connection_failure"
            raise OllamaClientError(category, reason) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaClientError("malformed_response", str(exc)) from exc
        except OSError as exc:
            raise OllamaClientError("connection_failure", str(exc)) from exc

        if not isinstance(data, dict):
            raise OllamaClientError("malformed_response", "response was not a JSON object")

        text = str(data.get("response", ""))
        elapsed = int((time.monotonic() - started) * 1000)
        return OllamaResponse(
            text=text,
            metrics={
                "latency_ms": elapsed,
                "prompt_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
            },
        )


def _http_error_category(status_code: int) -> str:
    if status_code == 401:
        return "cloud_authentication_required"
    if status_code == 404:
        return "model_not_found"
    if status_code == 413:
        return "context_overflow"
    if status_code == 429:
        return "rate_limit"
    return "http_error"
