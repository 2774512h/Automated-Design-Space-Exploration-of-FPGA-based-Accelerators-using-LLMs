import json
import os
import argparse
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(path: str) -> List[Dict]:
    
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f: 
            line = line.strip()
            if not line:
                continue 
            chunks.append(json.loads(line))
    return chunks 
    
def get_texts_from_chunks(chunks: List[Dict], use_field: str = "original_text") -> List[str]:
    texts = []
    for record in chunks:
        if use_field not in record:
            raise KeyError(f"Field '{use_field}' not found in record")
        texts.append(record[use_field])
        return texts
    
def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int =32, 
) -> np.ndarray:
    "Embed a list of texts using SentenceTransformer Model"
    "Return a NumPy array of shape (n_texts, embedding_dim)."
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts)} chunks")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Emdbeddings shape: {embeddings.shape}")
    return embeddings 

def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    chunks:List[Dict],
    embeddings_out: str,
    metadata_out: str,
):
    "Save embeddings to .npy and metadata to JSONL"

    emb_dir = os.path.dirname(embeddings_out)
    if emb_dir:
        os.makedirs(emb_dir, exist_ok=True)

    meta_dir = os.path.dirname(metadata_out)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)

    print(f"Saving embeddings to: {embeddings_out}")
    np.save(embeddings_out, embeddings)

    print(f"Saving metadata to: {metadata_out}")
    with open(metadata_out, "w", encoding = "utf-8") as f:
        for record in chunks:
            f.write(json.dumps(record) + "\n")

    print("Embeddings and metadata saved")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Vectorise textbook chunks using a sentence embedding model."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "--embeddings_out",
        "-e",
        default="data/index/textbook_embeddings.npy",
        help="Path to the output NumPy file for embeddings.",
    )
    parser.add_argument(
        "--metadata_out",
        "-m",
        default="data/index/textbook_metadata.jsonl",
        help="Path to the output JSONL file for metadata.",
    )
    parser.add_argument(
        "--model_name",
        "-s",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name.",
    )
    parser.add_argument(
        "--text_field",
        "-t",
        default="original_text",
        help="Which field of the JSONL to embed (original_text or lemmatised_text).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f"Loading chunks from: {args.input}")
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Using text field: {args.text_field}")
    texts = get_texts_from_chunks(chunks, use_field=args.text_field)

    embeddings = embed_texts(
        texts,
        model_name=args.model_name,
        batch_size=32,
    )

    save_embeddings_and_metadata(
        embeddings=embeddings,
        chunks=chunks,
        embeddings_out=args.embeddings_out,
        metadata_out=args.metadata_out,
    )


if __name__ == "__main__":
    main()  