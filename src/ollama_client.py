from __future__ import annotations

import os
from typing import Optional

import requests


def _get_host(host: Optional[str] = None) -> str:
    return (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")


def ollama_status(host: Optional[str] = None) -> tuple[bool, str]:
    """
    Check whether Ollama is reachable.
    """
    base = _get_host(host)
    url = f"{base}/api/tags"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return True, f"Ollama is available at {base}"
    except requests.RequestException as e:
        return (
            False,
            f"Ollama is not available at {base}. "
            f"Start Ollama with `ollama serve` or open the Ollama app. Error: {e}",
        )


def ollama_list_models(host: Optional[str] = None) -> list[str]:
    """
    Return installed Ollama model names from /api/tags.
    """
    base = _get_host(host)
    url = f"{base}/api/tags"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch Ollama models from {url}. "
            f"Is Ollama running? Error: {e}"
        ) from e

    models = data.get("models", [])
    names: list[str] = []

    for model in models:
        name = model.get("name")
        if name:
            names.append(name)

    return sorted(names)


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    base = _get_host(host)
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to call Ollama at {url}. Is Ollama running? "
            f"(Try: `ollama serve` or start the Ollama app.) Error: {e}"
        ) from e

    return (data.get("response") or "").strip()