from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ollama_client import ollama_list_models, ollama_status  # noqa: E402
from rag_service import get_default_prompt_prefix, run_rag_query  # noqa: E402

st.set_page_config(
    page_title="SECDA RAG-INTERFACE",
    page_icon="🧠",
    layout="wide",
)

st.title("SECDA RAG + Finetuned LLM")
st.caption("This interface allows you to experiment with the RAG Model for SECDA")

MODE_DESCRIPTIONS = {
    "baseline": "No retrieval. The model answers without retrieved SECDA context.",
    "rag_strict": "Retrieves SECDA chunks and answers using only retrieved context.",
    "rag_assisted": "Retrieves SECDA chunks and uses them as context",
}

RAG_MODES = {"rag_strict", "rag_assisted"}

with st.sidebar:
    st.header("Configuration")

    ollama_host = st.text_input(
        "Ollama host",
        value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    persist_dir = st.text_input(
        "Chroma persist directory",
        value="data/chroma",
    )

    collection_name = st.text_input(
        "Collection name",
        value="secda_docs",
    )

    retriever_model = st.text_input(
        "Retriever model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )

    prompt_mode = st.selectbox(
        "Prompt mode",
        options=["baseline", "rag_strict", "rag_assisted"],
        index=1,
        help="Choose which prompt mode to use.",
    )

    st.caption(MODE_DESCRIPTIONS[prompt_mode])

    if prompt_mode in RAG_MODES:
        n_results = st.slider(
            "Top-k retrieved chunks",
            min_value=1,
            max_value=10,
            value=5,
        )
    else:
        n_results = 5

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=64,
        max_value=1024,
        value=256,
        step=32,
    )

st.subheader("Model Selection")

ok, status_message = ollama_status(ollama_host)
if ok:
    st.info(status_message)
else:
    st.error(status_message)
    st.stop()

try:
    available_models = ollama_list_models(ollama_host)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if not available_models:
    st.warning("Ollama is running, but no installed models were found.")
    st.stop()

default_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
default_index = 0
if default_model in available_models:
    default_index = available_models.index(default_model)

selected_model = st.selectbox(
    "Installed Ollama models",
    options=available_models,
    index=default_index,
)

st.subheader("Prompt Setup")

default_prefix = get_default_prompt_prefix(prompt_mode)
customise_prompt = st.checkbox("Edit predefined prompt", value=False)

if customise_prompt:
    prompt_prefix = st.text_area(
        "Prompt prefix / system instruction",
        value=default_prefix,
        height=180,
    )
else:
    prompt_prefix = default_prefix
    st.code(default_prefix, language="text")

query = st.text_area(
    "Enter your prompt",
    height=180,
    placeholder="Ask a question about the SECDA documents...",
)

run_button = st.button("Run")

if run_button:
    if not query.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Building prompt, retrieving context if needed, and generating answer..."):
            try:
                result = run_rag_query(
                    query=query.strip(),
                    mode=prompt_mode,
                    ollama_host=ollama_host,
                    ollama_model=selected_model,
                    n_results=n_results,
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    retriever_model=retriever_model,
                    max_new_tokens=max_new_tokens,
                    prompt_prefix_override=prompt_prefix,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.subheader("Answer")
                st.write(result["answer"] or "_No answer returned._")

                st.subheader("Run Summary")
                st.write(f"**Mode:** {result.get('mode', prompt_mode)}")
                st.write(f"**Model:** {selected_model}")

                if result.get("mode") in RAG_MODES:
                    with st.expander("Retrieved context"):
                        context_items = result.get("context_items", [])
                        if not context_items:
                            st.info("No context chunks were retrieved.")
                        else:
                            for i, item in enumerate(context_items, start=1):
                                file_name = item.get("file", "unknown")
                                chunk_id = item.get("chunk_id", "unknown")
                                distance = item.get("distance")

                                header = f"Chunk {i} — file={file_name}, chunk_id={chunk_id}"
                                if distance is not None:
                                    try:
                                        header += f", distance={float(distance):.4f}"
                                    except Exception:
                                        pass

                                st.markdown(f"**{header}**")
                                st.write(item.get("text", ""))
                                st.markdown("---")

                with st.expander("Final prompt sent to the LLM"):
                    st.code(result.get("prompt", ""), language="text")