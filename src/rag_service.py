from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.llm import generate_answer
from src.query_rag import retrieve_context


DEFAULT_RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "secda_docs"
DEFAULT_PERSIST_DIR = "data/chroma"


PROMPT_PREFIXES = {
    "baseline": "You are a helpful assistant.",
    "rag_strict": (
        "You are an assistant answering questions about the SECDA design-space-exploration documents. "
        "Use the CONTEXT, which comes from the docs/ folder in the SECDA-TFLite repository, to answer "
        "the QUESTION. Base your answer only on the CONTEXT. Do not use outside knowledge. "
        "If the CONTEXT is insufficient, say that you do not know. "
        "Do not repeat the context verbatim. Answer concisely in 2-4 sentences."
    ),
    "rag_assisted": (
        "You are an assistant helping with SECDA design-space exploration and accelerator-design questions. "
        "Use the CONTEXT, retrieved from the SECDA-TFLite repository, as your primary evidence when answering "
        "the QUESTION. You may also use your general knowledge where it helps complete the answer, but clearly "
        "prioritise the retrieved context for project-specific details. If the retrieved context does not fully "
        "answer the question, explain what comes from the context and what is inferred or general knowledge. "
        "Do not repeat the context verbatim. Answer concisely in 2-4 sentences."
    ),
}


def baseline_prompt(query: str, prompt_prefix: Optional[str] = None) -> str:
    prefix = (
        prompt_prefix.strip()
        if prompt_prefix and prompt_prefix.strip()
        else PROMPT_PREFIXES["baseline"]
    )
    return f"{prefix}\n\nQuestion:\n{query}\n\nAnswer:"


def build_rag_context_block(
    context_items: List[Dict[str, Any]],
    max_context_chars: int = 6000,
) -> str:
    context_parts: List[str] = []
    total_chars = 0

    for item in context_items:
        text = item.get("text", "").strip()
        if not text:
            continue
        if total_chars + len(text) > max_context_chars:
            break
        context_parts.append(text)
        total_chars += len(text)

    return "\n\n".join(context_parts) if context_parts else "[NO CONTEXT RETRIEVED]"


def rag_prompt(
    query: str,
    mode: str,
    context_items: List[Dict[str, Any]],
    max_context_chars: int = 6000,
    prompt_prefix: Optional[str] = None,
) -> str:
    system_instruction = (
        prompt_prefix.strip()
        if prompt_prefix and prompt_prefix.strip()
        else PROMPT_PREFIXES[mode]
    )

    context_block = build_rag_context_block(
        context_items=context_items,
        max_context_chars=max_context_chars,
    )

    return (
        f"{system_instruction}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


def get_default_prompt_prefix(mode: str) -> str:
    try:
        return PROMPT_PREFIXES[mode]
    except KeyError:
        raise ValueError(f"Unknown mode: {mode}")


def run_rag_query(
    query: str,
    mode: str = "rag_strict",
    ollama_host: str = "http://localhost:11434",
    ollama_model: Optional[str] = None,
    n_results: int = 5,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    retriever_model: str = DEFAULT_RETRIEVER_MODEL,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    prompt_prefix_override: Optional[str] = None,
) -> Dict[str, Any]:
    context_items: List[Dict[str, Any]] = []

    if mode == "baseline":
        prompt = baseline_prompt(query, prompt_prefix_override)

    elif mode in {"rag_strict", "rag_assisted"}:
        context_items = retrieve_context(
            query=query,
            n_results=n_results,
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=retriever_model,
        )
        prompt = rag_prompt(
            query=query,
            mode=mode,
            context_items=context_items,
            prompt_prefix=prompt_prefix_override,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    answer = generate_answer(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        host=ollama_host,
        model=ollama_model,
        temperature=temperature,
    )

    return {
        "mode": mode,
        "query": query,
        "prompt": prompt,
        "context_items": context_items,
        "raw_context": "",
        "answer": answer,
    }