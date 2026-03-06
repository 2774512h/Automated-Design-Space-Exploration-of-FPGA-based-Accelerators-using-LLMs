import argparse
import json
import os
from typing import List, Dict

import spacy

# Global Chunking Pipeline
nlp = spacy.load("en_core_web_sm")
if "spacy_chunks" not in nlp.pipe_names:
    nlp.add_pipe("spacy_chunks", last=True, config={
        "chunking_method": "sentence",
        "chunk_size": 1,
        "overlap": 0,
        "truncate": False
    })
    
def load_text(path: str) -> str:
    """Load as a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_sentence_chunks(text:str) -> List[Dict]:
    """
    Split text into overlapping sentence chunks 
    """
    doc = nlp(text)
    
    chunks: List[Dict] = []

    for i, chunk in enumerate(doc._.chunks):
        chunk_text = " ".join(sent.text for sent in chunk)

        # First and last character to approximate char span
        start_char = chunk[0].start_char
        end_char = chunk[-1].end_char
        
        chunks.append({
            "id" : i,
            "original_text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
            # metadata can be added too
        })

    return chunks

def lemmatise_text(nlp, text: str) -> str:
    doc = nlp(text)
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct 
        and not token.is_space
        and not token.is_stop
    ]
    return " ".join(lemmas)

def spacy_pipeline(model_name: str = "en_core_web_sm"):
    """Load spaCy pipeline."""
    return spacy.load(model_name)

def load_docs_from_dir(root: str) -> List[Dict]:
    """
    Walk the SECDA docs directory and collect all text-like files.

    We only include files under `root` (docs folder), not README or tutorials.
    """
    records: List[Dict] = []

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            # Only keep markdown / text-like docs
            if not name.lower().endswith((".md", ".rst", ".txt")):
                continue

            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # Store relative path so we know which file each chunk came from
            records.append(
                {
                    "file": os.path.relpath(path, root),  # e.g. "usage.md"
                    "text": text,
                }
            )

    return records


def preprocess(
    input_path: str,
    output_path: str,
    spacy_model: str = "en_core_web_sm",
):
    print(f"Loading docs from: {input_path}")
    doc_files = load_docs_from_dir(input_path)
    print(f"Found {len(doc_files)} doc files")

    print("Splitting docs into chunks")
    all_chunks: List[Dict] = []
    next_id = 0

    for rec in doc_files:
        file_path = rec["file"]
        text = rec["text"]

        # Use your existing sentence-based chunker
        chunks = split_into_sentence_chunks(text)

        for c in chunks:
            c["id"] = next_id            # GLOBAL unique id across all docs
            c["file"] = file_path        # which doc this chunk came from
            next_id += 1
            all_chunks.append(c)

    print(f"Created {len(all_chunks)} chunks from {len(doc_files)} files.")

    print(f"Loading spaCy model: {spacy_model}")
    lemma_nlp = spacy_pipeline(spacy_model)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing processed chunks to: {output_path}")

    # Lemmatisation step over ALL chunks
    chunk_texts = [c["original_text"] for c in all_chunks]
    docs = lemma_nlp.pipe(chunk_texts, batch_size=16)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk, doc in zip(all_chunks, docs):
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if not token.is_punct
                and not token.is_space
                and not token.is_stop
            ]
            record = {
                **chunk,
                "lemmatised_text": " ".join(lemmas),
            }
            out_f.write(json.dumps(record) + "\n")

    print("Done.")



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess SECDA"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the SECDA directory",
)
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/secda_docs_chunks.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--spacy_model",
        "-s",
        default="en_core_web_sm",
        help="spaCy model name to use for lemmatisation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    preprocess(
        input_path=args.input,
        output_path=args.output,
        spacy_model=args.spacy_model,
    )

