import argparse
import textwrap

from llm import generate_answer
from query_rag import retrieve_context


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


def get_default_prompt_prefix(mode: str) -> str:
    try:
        return PROMPT_PREFIXES[mode]
    except KeyError:
        raise ValueError(f"Unknown mode: {mode}")


def build_context_block(context_items, max_context_chars: int = 6000) -> str:
    context_parts = []
    total_chars = 0

    for item in context_items:
        text = item["text"].strip()
        if not text:
            continue
        if total_chars + len(text) > max_context_chars:
            break
        context_parts.append(text)
        total_chars += len(text)

    return "\n\n".join(context_parts) if context_parts else "[NO CONTEXT RETRIEVED]"


def build_prompt(
    mode: str,
    query: str,
    context_items=None,
    max_context_chars: int = 6000,
) -> str:
    prefix = get_default_prompt_prefix(mode)

    if mode == "baseline":
        return (
            f"{prefix}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

    context_block = build_context_block(
        context_items or [],
        max_context_chars=max_context_chars,
    )

    return (
        f"{prefix}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run LLM experiments: baseline, rag_strict, rag_assisted."
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["baseline", "rag_strict", "rag_assisted"],
        required=True,
    )
    parser.add_argument("--query", "-q", required=True, help="User query string.")

    parser.add_argument(
        "--n_results",
        "-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (for RAG modes).",
    )
    parser.add_argument(
        "--persist_dir",
        "-p",
        default="data/chroma",
        help="Directory where Chroma index is stored (for RAG modes).",
    )
    parser.add_argument(
        "--collection_name",
        "-c",
        default="secda_docs",
        help="Chroma collection name (for RAG modes).",
    )
    parser.add_argument(
        "--retriever_model",
        "-rmodel",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name for RAG retrieval.",
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.mode == "baseline":
        prompt = build_prompt(args.mode, args.query)

    elif args.mode in ["rag_strict", "rag_assisted"]:
        context_items = retrieve_context(
            query=args.query,
            n_results=args.n_results,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.retriever_model,
        )

        print("\n=== Retrieved Context Chunks (RAG) ===\n")
        for i, item in enumerate(context_items, start=1):
            print(
                f"--- Chunk {i} "
                f"(file={item.get('file')}, chunk_id={item.get('chunk_id')}, "
                f"distance={item['distance']:.4f}) ---"
            )
            print(textwrap.fill(item["text"], width=100))
            print()

        prompt = build_prompt(args.mode, args.query, context_items)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    text = generate_answer(prompt)

    print("\n=== Model Output ===\n")
    print(text)
    