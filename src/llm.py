from __future__ import annotations

import os
from typing import Optional

import requests

_DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
_DEFAULT_TEMPERATURE = os.getenv("OLLAMA_TEMPERATURE", "0.0")


def _get_temperature(temperature: Optional[float]) -> float:
    if temperature is not None:
        return float(temperature)
    try:
        return float(_DEFAULT_TEMPERATURE)
    except ValueError:
        return 0.0


def load_model(host: Optional[str] = None, timeout: int = 10) -> None:
    """
    For Ollama, there's no model object to load into Python.
    This simply checks the server is reachable.
    """
    host = (host or _DEFAULT_HOST).rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Is Ollama running? "
            f"(Start the Ollama app or run `ollama serve`.) Error: {e}"
        ) from e


def generate_answer(
    prompt: str,
    max_new_tokens: int = 256,
    host: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Generate an answer via Ollama and return ONLY the text.
    """
    host = (host or _DEFAULT_HOST).rstrip("/")
    model = model or _DEFAULT_MODEL
    temperature = _get_temperature(temperature)

    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": int(max_new_tokens),
            "temperature": temperature,
        },
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    
    print(f"[LLM] Using Ollama model={model} host={host}")
    return (data.get("response") or "").strip()