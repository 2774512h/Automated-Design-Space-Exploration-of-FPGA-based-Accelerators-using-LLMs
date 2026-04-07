from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import chromadb


def list_collection_names(persist_dir: str) -> Tuple[bool, List[str], str]:
    persist_path = Path(persist_dir)

    if not persist_path.exists():
        return (
            False,
            [],
            f"SECDA index not found. The directory '{persist_dir}' does not exist yet.",
        )

    try:
        client = chromadb.PersistentClient(path=str(persist_path))
    except Exception as exc:
        return (
            False,
            [],
            f"Could not open Chroma persist directory '{persist_dir}': {exc}",
        )

    try:
        collections = client.list_collections()
    except Exception as exc:
        return (
            False,
            [],
            f"Opened '{persist_dir}', but could not list Chroma collections: {exc}",
        )

    names: List[str] = []
    for collection in collections:
        name = getattr(collection, "name", None)
        if isinstance(name, str) and name.strip():
            names.append(name)

    return True, names, f"Found {len(names)} collection(s) in '{persist_dir}'."


def check_secda_index_ready(
    persist_dir: str,
    collection_name: str,
) -> Tuple[bool, str]:
    ok, collection_names, message = list_collection_names(persist_dir)

    if not ok:
        return False, message

    if not collection_names:
        return (
            False,
            f"Chroma directory '{persist_dir}' exists, but it contains no collections yet.",
        )

    if collection_name not in collection_names:
        available = ", ".join(sorted(collection_names))
        return (
            False,
            f"Collection '{collection_name}' was not found in '{persist_dir}'. "
            f"Available collection(s): {available}",
        )

    try:
        client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        collection = client.get_collection(collection_name)
    except Exception as exc:
        return (
            False,
            f"Collection '{collection_name}' appears to exist in '{persist_dir}', "
            f"but could not be opened: {exc}",
        )

    try:
        count = collection.count()
    except Exception as exc:
        return (
            False,
            f"Collection '{collection_name}' exists, but its document count could not be read: {exc}",
        )

    if count <= 0:
        return (
            False,
            f"Collection '{collection_name}' exists in '{persist_dir}', but it is empty.",
        )

    return (
        True,
        f"SECDA index is ready.",
    )