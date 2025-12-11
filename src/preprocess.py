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
        "chunk_size": 2,
        "overlap": 1,
        "truncate": False
    })
    
def load_text(path: str) -> str:
    """Load the full textbook as a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_sentence_chunks(text:str) -> List[Dict]:
    """
    Split text into overlapping sentence chunks """
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


def preprocess(
    input_path: str,
    output_path: str,
    spacy_model: str = "en_core_web_sm",
):
    print(f"Loading textbook from: {input_path}")
    text = load_text(input_path)

    print(f"Splitting text into chunks")
    chunks = split_into_sentence_chunks(text)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading spaCy model: {spacy_model}")
    lemma_nlp = spacy_pipeline(spacy_model)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing processed chunks to: {output_path}")

    chunk_texts = [c["original_text"] for c in chunks]
    docs = lemma_nlp.pipe(chunk_texts, batch_size=16)
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk, doc in zip(chunks, docs):
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
        description="Preprocess a CS textbook: chunk + lemmatisation"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the raw textbook text file (e.g. data/raw/textbook.txt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/textbook_chunks.jsonl",
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

