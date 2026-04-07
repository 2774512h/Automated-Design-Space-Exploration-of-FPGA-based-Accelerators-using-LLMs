from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from rag_service import run_rag_query

BATCH_MODES = ["baseline", "rag_strict", "rag_assisted"]


def parse_questions_from_txt(text: str) -> List[str]:
    """Return one question per non-empty line."""
    questions: List[str] = []
    for line in text.splitlines():
        question = line.strip()
        if question:
            questions.append(question)
    return questions


def run_batch_queries(
    questions: Iterable[str],
    *,
    ollama_host: str,
    ollama_model: Optional[str],
    n_results: int,
    persist_dir: str,
    collection_name: str,
    retriever_model: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    prompt_prefix_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    batch_started_at = datetime.now(timezone.utc).isoformat()

    for question_id, question in enumerate(questions, start=1):
        for mode in BATCH_MODES:
            prompt_override = None
            if prompt_prefix_overrides:
                prompt_override = prompt_prefix_overrides.get(mode)

            result = run_rag_query(
                query=question,
                mode=mode,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                n_results=n_results,
                persist_dir=persist_dir,
                collection_name=collection_name,
                retriever_model=retriever_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_prefix_override=prompt_override,
            )

            context_items = result.get("context_items", []) or []
            context_files = sorted(
                {
                    str(item.get("file", "")).strip()
                    for item in context_items
                    if str(item.get("file", "")).strip()
                }
            )

            rows.append(
                {
                    "batch_started_at_utc": batch_started_at,
                    "question_id": question_id,
                    "question": question,
                    "mode": mode,
                    "answer": result.get("answer", ""),
                    "model": ollama_model or "",
                    "ollama_host": ollama_host,
                    "n_results": n_results,
                    "persist_dir": persist_dir,
                    "collection_name": collection_name,
                    "retriever_model": retriever_model,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "retrieved_context_count": len(context_items),
                    "retrieved_files": " | ".join(context_files),
                    "context_char_count": result.get("context_char_count", 0),
                    "answer_word_count": result.get("answer_word_count", 0),
                    "retrieval_latency_seconds": result.get("retrieval_latency_seconds", 0.0),
                    "generation_latency_seconds": result.get("generation_latency_seconds", 0.0),
                    "total_latency_seconds": result.get("total_latency_seconds", 0.0),
                }
            )

    return rows


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")