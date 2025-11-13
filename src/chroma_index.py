import argparse
import json 
import os
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

def load_chunks(path: str) -> List[Dict]:
    
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f: 
            line = line.strip()
            if not line:
                continue 
            chunks.append(json.loads(line))
    return chunks 

def build_chroma_collection(
        chunks: List[Dict],
        persist_dir: str = "data/chroma",
        collection_name: str = "textbook",
        model_name: str = "sentnece-transformers/all-MiniLM-L6-v2",
):

    "Create a Chroma collection and add all chunks"
    "Chroma calls SentenceTransformers for under hood embeddings"

    os.makedirs(persist_dir, exist_ok=True)

    print(f"Using Chroma persist directory: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)

    #sentence transformers as the embedding function
    print(f"Using embedding model: {model_name}")
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    try: 
        client.delete_collection(name=collection)
        print(f"Deleted existing collection if it existed")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=st_ef
    )

    #prepare data
    ids=[]
    docs=[]
    metadatas=[]

    for record in chunks:
        rid = str(record['id'])
        text = record['original_text']
        ids.append(rid)
        docs.append(text)
        metadatas.append(
            {
                'id': record['id']
                #any metadata fields to be added
            }
        )
    print(f"Adding {len(ids)} chunks to collection '{collection_name}'...")
    collection.add(ids=ids, documents=docs, metadatas=metadatas)
    print("Index build complete")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB index from textbook chunks JSONL."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the JSONL file with chunks (e.g. data/processed/textbook_chunks.jsonl).",
    )
    parser.add_argument(
        "--persist_dir",
        "-p",
        default="data/chroma",
        help="Directory where Chroma will persist the index.",
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
        help="SentenceTransformers model to use for embeddings.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f"Loading chunks from: {args.input}")
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks.")

    build_chroma_collection(
        chunks=chunks,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()