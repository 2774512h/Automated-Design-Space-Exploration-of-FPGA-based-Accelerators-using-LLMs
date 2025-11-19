import argparse
import textwrap

from llm import generate_answer
from query_rag import retrieve_context

def baseline_prompt(query: str) -> str:
    return(
        "You are a helpful assistant.\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

def context_prompt(query: str, context: str) -> str:
    return(
        "You are an assistant that answers questions using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

def rag_prompt(
    query: str,
    context_items,
    max_context_chars: int = 6000,
) -> str: 
    system_instruction = (
        ""
    )
    
    context_parts = [] # List of text chunks 
    total_chars=0

    for item in context_items:
        text = item["text"].strip()
        if not text:
            continue
        if total_chars + len(text) > max_context_chars:
            break
        context_parts.append(text)
        total_chars += len(text)

    context_block = "\n\n".join(context_parts) if context_parts else "[NO CONTEXT RETRIEVED]"

    prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )
    return prompt

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run LLM experiments: baseline, context, rag."
    )
    parser.add_argument("--mode", "-m", choices=["baseline", "context", "rag"], required=True)
    parser.add_argument("--query", "-q", required=True, help="User query string.")

    #context mode
    parser.add_argument(
        "--context_file",
        "-cfile",
        default="data/context.txt",
        help="Path to context file (for mode=context).",
    )

    #RAG mode
    parser.add_argument(
        "--n_results",
        "-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (for mode=rag).",
    )
    parser.add_argument(
        "--persist_dir",
        "-p",
        default="data/chroma",
        help="Directory where Chroma index is stored (for mode=rag).",
    )
    parser.add_argument(
        "--collection_name",
        "-c",
        default="textbook",
        help="Chroma collection name (for mode=rag).",
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

    # Build prompt based on the mode
    
    if args.mode == "baseline":
        prompt = baseline_prompt(args.query)

    elif args.mode == "context":
        with open(args.context_file, "r", encoding="utf-8") as f:
            context = f.read()
        prompt = context_prompt(args.query, context)

    elif args.mode == "rag":
        context_items = retrieve_context(
            query=args.query,
            n_results=args.n_results,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.retriever_model,
        )

        # Print to the console to inspect what's being fed to the model
        print("\n=== Retrieved Context Chunks (RAG) ===\n")
        for i, item in enumerate(context_items, start=1):
            print(f"--- Chunk {i} (id={item['id']}, distance={item['distance']:.4f}) ---")
            print(textwrap.fill(item["text"], width=100))
            print()

        prompt = rag_prompt(args.query, context_items)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Call the LLM
    text, num_tokens, elapsed, tps = generate_answer(prompt)

    print("\n=== Model Output ===\n")
    print(text)

    print(f"\n[METRICS] mode={args.mode}, tokens={num_tokens}, time={elapsed:.3f}s, tps={tps:.2f}")


if __name__ == "__main__":
    main()
    