import os
import chromadb

PERSIST_DIR = os.getenv("CHROMA_DIR", "data/chroma_secda")
COLLECTION = os.getenv("COLLECTION", "secda_docs")

client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_collection(COLLECTION)

print("PERSIST_DIR:", PERSIST_DIR)
print("COLLECTION:", COLLECTION)

print("COUNT:", col.count())

peek = col.peek(limit=5)
print("\nPEEK METADATA EXAMPLE:")
if peek.get("metadatas") and len(peek["metadatas"]) > 0:
    print(peek["metadatas"][0])
else:
    print("No metadatas returned (or empty).")

print("\nPEEK DOCUMENT EXAMPLE:")
if peek.get("documents") and len(peek["documents"]) > 0:
    print(peek["documents"][0][:300], "...")