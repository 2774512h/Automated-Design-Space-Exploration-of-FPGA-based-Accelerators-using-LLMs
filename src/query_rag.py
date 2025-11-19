import argparse
import textwrap

import chromadb
from chromadb.utils import embedding_functions

# Connect to existing ChromaDB collection
def get_collection(
    persist_dir: str = "data/chroma",
    collection_name: str = "textbook",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    client = chromadb.PersistentClient(path=persist_dir)
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    collection = client.get_collection(
        name=collection_name,
        embedding_function=st_ef
    )
    return collection

def retrieve_context(
    query: str,
    n_results: int,
    persist_dir: str,
    collection_name: str,
    model_name: str,
):
    # Retrieval function
    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )

    # Embed query and find n most similar chunks 
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    docs = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[]])[0]

    context_items = []

    # Build a list of dictionaries 
    for doc_id, doc_text, dist, meta in zip(ids, docs, distances, metadatas):
        context_items.append(
            {
                "id": doc_id,
                "text": doc_text,
                "distance": dist,
                "metadata": meta,
            }
        )
    return context_items

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query a ChromaDB collection."
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="User query string.",
    )
    parser.add_argument(
        "--n_results",
        "-k",
        type=int,
        default=5,
        help="Number of results to retrieve.",
    )
    parser.add_argument(
        "--persist_dir",
        "-p",
        default="data/chroma",
        help="Directory where Chroma index is stored.",
    )
    parser.add_argument(
        "--collection_name",
        "-c",
        default="textbook",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name (must match index).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    context_items=retrieve_context(
        query=args.query,
        n_results=args.n_results,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
    )
    print("\n=== Retrieved Context Chunks ===\n")
    for i, item in enumerate(context_items, start=1):
        print(f"--- Chunk {i} (id={item['id']}, distance={item['distance']:.4f}) ---")
        print(textwrap.fill(item["text"], width=100))
        print()


if __name__ == "__main__":
    main()