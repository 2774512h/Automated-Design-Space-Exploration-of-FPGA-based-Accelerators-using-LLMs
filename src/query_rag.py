import argparse
import textwrap
import json
import re

import chromadb
from chromadb.utils import embedding_functions

from typing import List, Dict, Any

# Maybe add as param for more modularity?
CHUNKS_PATH = "data/processed/secda_docs_chunks.jsonl"
LEXICAL_WEIGHT = 1.0

# Maybe look to add retrival_utils later for shared functions

def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    return set(tokens)

def lexical_score(query: str, doc: str) -> float:
    query_tokens = tokenize(query)
    document_tokens = tokenize(doc)

    if not query_tokens:
        return 0.0
    
    intersection = query_tokens.intersection(document_tokens)
    return len(intersection) / len(query_tokens)

def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

CHUNKS: List[Dict[str, Any]] = load_chunks(CHUNKS_PATH)
CHUNKS_BY_FILE_AND_ID: Dict[tuple[str, int], Dict[str, Any]] = {}
for rec in CHUNKS:
    f = rec.get("file", "unknown")
    cid = int(rec["id"])
    CHUNKS_BY_FILE_AND_ID[(f, cid)] = rec

# Connect to existing ChromaDB collection
def get_collection(
    persist_dir: str = "data/chroma",
    collection_name: str = "secda_docs",
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

    n_candidates = max(30, n_results * 3)

    # Embed query and find n most similar chunks 
    results = collection.query(
        query_texts=[query],
        n_results=n_candidates,
        include=["documents", "distances", "metadatas"],
    )

    docs = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[]])[0]

    candidates: List[Dict[str, Any]] = []

    # Build a list of dictionaries 
    for doc_id, doc_text, dist, meta in zip(ids, docs, distances, metadatas):
        meta = meta if isinstance(meta, dict) else {}
        file_name = meta.get("file", "unknown")
        chunk_id = meta.get("chunk_id", None)

        # Fallback: parse "file:chunk" if chunk_id missing
        if chunk_id is None:
            try:
                chunk_id = int(str(doc_id).split(":")[-1])
            except Exception:
                # last resort: skip
                continue

        candidates.append(
            {
                "file": file_name,
                "chunk_id": int(chunk_id),
                "doc_id": doc_id,
                "text": doc_text,
                "distance": float(dist),
                "metadata": meta,
            }
        )
    
    for candidate in candidates:
        sem_score = -candidate["distance"]
        lex_score = lexical_score(query, candidate["text"])
        candidate["sem_score"] = sem_score
        candidate["lex_score"] = lex_score
        candidate["score"] = sem_score + LEXICAL_WEIGHT * lex_score

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[:n_results]

    # add id-1 and id+1
    selected: Dict[tuple[str, int], Dict[str, Any]] = {}

    for c in top:
        key = (c["file"], c["chunk_id"])
        selected[key] = c

    for (f, cid) in list(selected.keys()):
        for offset in (-1, 1):
            nid = cid + offset
            nkey = (f, nid)
            if nkey in CHUNKS_BY_FILE_AND_ID and nkey not in selected:
                rec = CHUNKS_BY_FILE_AND_ID[nkey]
                selected[nkey] = {
                    "file": f,
                    "chunk_id": nid,
                    "doc_id": f"{f}:{nid}",
                    "text": rec["original_text"],
                    "distance": selected[(f, cid)]["distance"],  # reuse parent distance
                    "metadata": rec,
                }

    context_items = list(selected.values())
    context_items.sort(key=lambda c: (c["distance"], c["file"], c["chunk_id"]))
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
        default="secda_docs",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name (must match index).",
    )
    return parser.parse_args()

    return context_items_reranked
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
        print(f"--- Chunk {i} (file={item.get('file')}, id={item.get('chunk_id')}, distance={item['distance']:.4f}) ---")
        print(textwrap.fill(item["text"], width=100))
        print()


if __name__ == "__main__":
    main()