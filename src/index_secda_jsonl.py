import argparse
import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DEFAULT_JSONL_PATH = os.getenv("SECDA_JSONL", "data/processed/secda_docs_chunks.jsonl")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_DIR", "data/chroma_secda")
DEFAULT_COLLECTION = os.getenv("COLLECTION", "secda_docs_v1")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Chroma collection from SECDA JSONL chunks."
    )
    parser.add_argument(
        "--jsonl_path",
        default=DEFAULT_JSONL_PATH,
        help="Path to the processed SECDA JSONL file.",
    )
    parser.add_argument(
        "--persist_dir",
        default=DEFAULT_PERSIST_DIR,
        help="Directory where the Chroma database should be stored.",
    )
    parser.add_argument(
        "--collection_name",
        default=DEFAULT_COLLECTION,
        help="Name of the Chroma collection to create.",
    )
    parser.add_argument(
        "--embed_model",
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformers embedding model name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Cannot find {jsonl_path.resolve()}")

    persist_dir = args.persist_dir
    collection_name = args.collection_name
    embed_model = args.embed_model

    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"source": "secda_docs_chunks"},
    )

    documents = []
    metadatas = []
    ids = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc = obj.get("original_text", "")
            file_name = obj.get("file", "unknown")
            chunk_id = obj.get("id")

            uid = f"{file_name}:{chunk_id}"

            documents.append(doc)
            metadatas.append(
                {
                    "file": file_name,
                    "chunk_id": int(chunk_id) if chunk_id is not None else None,
                    "start_char": obj.get("start_char"),
                    "end_char": obj.get("end_char"),
                }
            )
            ids.append(uid)

    batch_size = 64
    for i in range(0, len(ids), batch_size):
        col.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print(
        f"Indexed {col.count()} chunks into '{persist_dir}' / '{collection_name}' "
        f"using model '{embed_model}'."
    )


if __name__ == "__main__":
    main()