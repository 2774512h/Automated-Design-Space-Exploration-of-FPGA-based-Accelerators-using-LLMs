import argparse
import textwrap
from tree_retrieval import retrieve_with_tree

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-k", "--n_results", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    chunks = retrieve_with_tree(
        question=args.query,
        n_results=args.n_results,
        neighbor_window=1,
        n_candidates_per_group=30,
    )

    print("\n=== Tree-RAG Retrieved Context Chunks ===\n")
    for i, c in enumerate(chunks, start=1):
        print(f"--- Chunk {i} (id={c['id']}, distance={c['distance']:.4f}) ---")
        print(textwrap.fill(c["text"], width=100))
        print()

if __name__ == "__main__":
    main()
