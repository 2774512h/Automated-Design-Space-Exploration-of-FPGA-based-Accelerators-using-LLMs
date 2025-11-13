import argparse
import textwrap

import chromadb
from chromadb.utils import embedding_functions

def get_collection(
    persist_dir: str = "data/chroma",
    collection_name: str = "textbook",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    #connect to an existing collection with an embedding function
    client = chromadb.PersistentClient(path=persist_dir)
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    collection = client.get_collection(
        name=collection_name,
        embedding_function=st_ef
    )
    return collection

def query_collection(
    query: str,
    n_results: int = 5,
    persist_dir: str = "textbook",
    collection_name: str = "textbook",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )

    print(f"\nQuery: {query}\n")
    results = collection.query(
        query_texts=[query],
        n_results = n_results,
    )

    #results is a dict with the keys: ids, docs, metadatas, distances
    docs = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[]])[0]
    
    for rank, (doc_id, doc_text, dist, meta) in enumerate(
        zip(ids, docs, distances, metadatas), start=1
    ):
        print(f"=== Result {rank} ===")
        print(f"Chunk ID: {doc_id}")
        if meta:
            print(f"Metadata: {meta}")
        print(f"Distance: {dist:.4f}")
        print("Text:")
        print(textwrap.fill(doc_text, width=100))
        print()
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query a ChromaDB collection built from the textbook."
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
    query_collection(
        query=args.query,
        n_results=args.n_results,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()