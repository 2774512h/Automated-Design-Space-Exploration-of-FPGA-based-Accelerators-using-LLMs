import argparse
import json
import os
from typing import List

import spacy


def load_text(path: str) -> str:
    """Load the full textbook as a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, max_words: int = 250) -> List[str]:
    """Split text into roughly max_words-sized chunks."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

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


def preprocess_textbook(
    input_path: str,
    output_path: str,
    max_words_per_chunk: int = 250,
    spacy_model: str = "en_core_web_sm",
):
    print(f"Loading textbook from: {input_path}")
    text = load_text(input_path)

    print(f"Splitting text into chunks of ~{max_words_per_chunk} words...")
    chunks = split_into_chunks(text, max_words=max_words_per_chunk)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading spaCy model: {spacy_model}")
    nlp = spacy_pipeline(spacy_model)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing processed chunks to: {output_path}")
    docs = nlp.pipe(chunks, batch_size=16)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, (chunk, doc) in enumerate(zip(chunks, docs)):
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if not token.is_punct
                and not token.is_space
                and not token.is_stop
            ]
            record = {
                "id": i,
                "original_text": chunk,
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
        "--max_words",
        "-m",
        type=int,
        default=250,
        help="Approximate number of words per chunk.",
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
    preprocess_textbook(
        input_path=args.input,
        output_path=args.output,
        max_words_per_chunk=args.max_words,
        spacy_model=args.spacy_model,
    )

