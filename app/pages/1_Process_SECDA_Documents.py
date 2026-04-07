from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secda_index_utils import check_secda_index_ready

DEFAULT_PERSIST_DIR = "data/chroma_secda"
DEFAULT_COLLECTION_NAME = "secda_docs_v1"
DEFAULT_JSONL_PATH = "data/processed/secda_docs_chunks.jsonl"

PREPROCESS_SCRIPT = SRC_DIR / "preprocess.py"
INDEX_SCRIPT = SRC_DIR / "index_secda_jsonl.py"

persist_dir = DEFAULT_PERSIST_DIR
collection_name = DEFAULT_COLLECTION_NAME

st.set_page_config(
    page_title="Process SECDA Documents",
    page_icon="🛠️",
    layout="wide",
)

st.title("Process SECDA Documents")
st.caption(
    "Use this page to preprocess a local SECDA-TFLite repository and build the Chroma index used by the RAG interface."
)

st.markdown(
    """
Before using RAG modes, you need to build the SECDA index locally.

Enter the **root directory** of your local SECDA-TFLite repository exactly as it appears on this machine.
This must be a local folder path, not a GitHub URL.

Running the full build will overwrite the existing SECDA collection with the same fixed name.
"""
)

st.subheader("Fixed SECDA Index Configuration")
st.code(f"Persist directory: {persist_dir}", language="text")
st.code(f"Collection name: {collection_name}", language="text")

st.subheader("Current SECDA Index Status")

jsonl_path = DEFAULT_JSONL_PATH
st.code(f"Processed JSONL path: {jsonl_path}", language="text")

ready, message = check_secda_index_ready(
    persist_dir=persist_dir,
    collection_name=collection_name,
)

if ready:
    st.success(message)
else:
    st.warning(message)

if st.button("Refresh SECDA Index Status", use_container_width=True):
    ready, message = check_secda_index_ready(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    if ready:
        st.success(message)
    else:
        st.warning(message)

st.subheader("Local SECDA Repository")

secda_root = st.text_input(
    "SECDA-TFLite root directory on this machine",
    placeholder=r"Example: C:\Users\you\Documents\SECDA-TFLite",
)

show_advanced = st.checkbox("Show script paths", value=False)
if show_advanced:
    st.code(f"Preprocess script: {PREPROCESS_SCRIPT}", language="text")
    st.code(f"Index script: {INDEX_SCRIPT}", language="text")

st.subheader("Build Tools")

col1, col2 = st.columns(2)

with col1:
    run_preprocess = st.button("Run Preprocess", use_container_width=True)

with col2:
    run_index = st.button("Build Chroma Index", use_container_width=True)

run_both = st.button("Run Full SECDA Build", type="primary", use_container_width=True)


def run_command(command: list[str], label: str) -> bool:
    st.markdown(f"### {label}")
    st.code(" ".join(f'"{part}"' if " " in part else part for part in command), language="bash")

    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        st.error(f"{label} failed to start: {exc}")
        return False

    if completed.stdout:
        st.markdown("**Standard output**")
        st.code(completed.stdout, language="text")

    if completed.stderr:
        st.markdown("**Standard error**")
        st.code(completed.stderr, language="text")

    if completed.returncode != 0:
        st.error(f"{label} failed with exit code {completed.returncode}.")
        return False

    st.success(f"{label} completed successfully.")
    return True


def validate_secda_root(path_text: str) -> Path | None:
    if not path_text.strip():
        st.error("Please enter the local SECDA-TFLite root directory first.")
        return None

    root = Path(path_text.strip())

    if not root.exists():
        st.error(f"The path '{root}' does not exist on this machine.")
        return None

    if not root.is_dir():
        st.error(f"The path '{root}' is not a directory.")
        return None

    return root


def preprocess_command(root: Path) -> list[str]:
    return [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--input",
        str(root),
        "--output",
        jsonl_path,
    ]


def index_command() -> list[str]:
    return [
        sys.executable,
        str(INDEX_SCRIPT),
        "--jsonl_path",
        jsonl_path,
        "--persist_dir",
        persist_dir,
        "--collection_name",
        collection_name,
    ]


if run_preprocess:
    root = validate_secda_root(secda_root)
    if root is not None:
        with st.spinner("Running preprocessing..."):
            run_command(
                preprocess_command(root),
                "Preprocess SECDA Documents",
            )

if run_index:
    with st.spinner("Building Chroma index..."):
        run_command(
            index_command(),
            "Build SECDA Chroma Index",
        )

if run_both:
    root = validate_secda_root(secda_root)
    if root is not None:
        ok_pre = False
        ok_idx = False

        with st.spinner("Running full SECDA build..."):
            ok_pre = run_command(
                preprocess_command(root),
                "Preprocess SECDA Documents",
            )

            if ok_pre:
                ok_idx = run_command(
                    index_command(),
                    "Build SECDA Chroma Index",
                )

        if ok_pre and ok_idx:
            st.subheader("Updated SECDA Index Status")
            ready, message = check_secda_index_ready(
                persist_dir=persist_dir,
                collection_name=collection_name,
            )
            if ready:
                st.success(message)
            else:
                st.warning(message)