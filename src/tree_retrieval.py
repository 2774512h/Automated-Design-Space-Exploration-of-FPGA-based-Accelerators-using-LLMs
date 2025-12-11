import json
import re
from typing import List, Dict, Any, Set, Tuple

import chromadb
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

import numpy as np

CHUNKS_PATH = "data/processed/textbook_chunks_grouped.jsonl"
PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "textbook"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LEXICAL_WEIGHT = 1.0   
TOP_GROUPS = 3         

def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

CHUNKS: List[Dict[str, Any]] = load_chunks(CHUNKS_PATH)

# Dictionary mapping id to record
CHUNKS_BY_ID: Dict[int, Dict[str, Any]] = {
    int(rec["id"]): rec for rec in CHUNKS
}

# Dictionary k: group_id v: list of all chunks records in group
GROUPS: Dict[int, List[Dict[str, Any]]] = {}

# Later swap to file, folder etc
for rec in CHUNKS:
    gid = int(rec.get("group_id", 0))
    GROUPS.setdefault(gid, []).append(rec)

# Exists in Query rag
def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    return set(tokens)

# Exists in Query rag
def lexical_overlap_score(query: str, doc: str) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    if not q_tokens:
        return 0.0
    inter = q_tokens.intersection(d_tokens)
    return len(inter) / len(q_tokens)

# Sentence transformer
GROUP_MODEL = SentenceTransformer(EMBEDDING_MODEL)

# Connect to Chroma, set embed fnctn with same model as index
client = chromadb.PersistentClient(path=PERSIST_DIR)
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# Stores chunk embeddings
CHUNK_COLLECTION = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function = st_ef,
) # Chroma wrapper

# Holds text and embedding vec
GROUP_NODES: Dict[int, Dict[str, Any]] = {}

def build_group_nodes():
    texts: List[str] = []
    gids: List[int] = []

    for gid, chunks in GROUPS.items():
        # Sort chunks by id
        chunks_sorted = sorted(chunks, key=lambda r: int(r["id"]))

        # Take the first 3 chunks and join original text
        sample_text = " ".join(
            rec["original_text"] for rec in chunks_sorted[:3]
        )

        texts.append(sample_text)
        gids.append(gid)

    # Embed all group texts with the model
    embeddings = GROUP_MODEL.encode(texts, batch_size=16, show_progress_bar=False)
    for gid, emb, text in zip(gids, embeddings, texts):
        # Store
        GROUP_NODES[gid] = {
            "text": text,
            "embedding": emb,
        }

build_group_nodes()

def retrieve_top_groups(question: str, top_k: int = TOP_GROUPS) -> List[Tuple[int, float]]:
    """
    Returns list of (group_id, score) sorted by score desc.
    """
    # Embed the question once 
    q_emb = GROUP_MODEL.encode([question], show_progress_bar=False)[0]

    scored: List[Tuple[int, float]] = []
    for gid, node in GROUP_NODES.items():
        emb = node["embedding"]
        # cosine similarity
        sem = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
        lex = lexical_overlap_score(question, node["text"])
        score = sem + LEXICAL_WEIGHT * lex
        scored.append((gid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def retrieve_with_tree(
    question: str,
    n_results: int = 5,
    neighbor_window: int = 1,
    n_candidates_per_group: int = 30,
) -> List[Dict[str, Any]]:
    """
    Tree-based retrieval:
    1. Coarse: choose top groups.
    2. Fine: within those groups, choose best chunks using the
       existing Chroma + lexical scoring.
    3. Neighbor expansion by id +/- 1.
    """
    # Run group-level retriever 
    top_groups = retrieve_top_groups(question, top_k=TOP_GROUPS)
    
    # Take just the group ids 
    selected_group_ids = [gid for gid, _ in top_groups]

    # Gather chunk IDs from any top selected groups 
    candidate_ids: List[int] = []
    for gid in selected_group_ids:
        for rec in GROUPS[gid]:
            candidate_ids.append(int(rec["id"]))
    
    # Get broad semantic candidate pool
    results = CHUNK_COLLECTION.query(
        query_texts=[question],
        n_results=max(50, n_results * 4),
        include=["documents", "metadatas", "distances"],
    )

    # Unpack first query's results (only one)
    docs = results["documents"][0]
    ids = results["ids"][0]
    dists = results["distances"][0]
    metadatas = results.get("metadatas", [[]])[0]

    candidates: List[Dict[str, Any]] = []

    for doc_id, doc_text, dist, meta in zip(ids, docs, dists, metadatas):
        # Original id from metadata
        rec_id = meta.get("id") if isinstance(meta, dict) else None
        
        # Fallback 
        if rec_id is None:
            try:
                rec_id = int(doc_id)
            except Exception:
                continue

        rec_id = int(rec_id)

        # Not in selected groups
        if rec_id not in candidate_ids:
            continue

        # smaller distance = better score 
        sem_score = -float(dist)
        lex_score = lexical_overlap_score(question, doc_text)
        score = sem_score + LEXICAL_WEIGHT * lex_score

        candidates.append(
            {
                "id": rec_id,
                "text": doc_text,
                "distance": float(dist),
                "metadata": meta,
                "sem_score": sem_score,
                "lex_score": lex_score,
                "score": score,
            }
        )

    # if nothing survives, fall back to first n_results from raw candidates
    if not candidates:
        return []

    # sort by combined score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[:n_results]

    # Neighbour expansion using CHUNKS_BY_ID
    selected: Dict[int, Dict[str, Any]] = {c["id"]: c for c in top}

    for cid, parent in list(selected.items()):
        for offset in range(-neighbor_window, neighbor_window + 1):
            if offset == 0:
                continue
            nid = cid + offset

            # If in CHUNKS and not selected already 
            if nid in CHUNKS_BY_ID and nid not in selected:
                rec = CHUNKS_BY_ID[nid]
                # Fetch original text 
                text = rec["original_text"]
                # Same semantic score 
                sem_score = parent["sem_score"]
                # Compute lexical score 
                lex_score = lexical_overlap_score(question, text)
                # Evaluate score 
                score = sem_score + LEXICAL_WEIGHT * lex_score

                # Add
                selected[nid] = {
                    "id": nid,
                    "text": text,
                    "distance": parent["distance"],
                    "metadata": rec,
                    "sem_score": sem_score,
                    "lex_score": lex_score,
                    "score": score,
                }

    # Final list 
    items = list(selected.values())
    items.sort(key=lambda c: (c["score"], -c["sem_score"], c["id"]), reverse=True)
    return items

# Turns into a string for the LLM 
def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for c in chunks:
        gid = CHUNKS_BY_ID[c["id"]].get("group_id", "NA")
        parts.append(f"[Chunk {c['id']} | group {gid}]\n{c['text']}")
    return "\n\n".join(parts)


