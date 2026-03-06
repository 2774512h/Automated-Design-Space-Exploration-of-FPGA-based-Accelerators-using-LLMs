# src/ollama_client.py
from __future__ import annotations

import os
from typing import Optional

import requests


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # simplest: return once
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

    # Ollama returns {"response": "...", ...}
    return (data.get("response") or "").strip()