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

from ollama_client import ollama_list_models, ollama_status
from batch_eval import parse_questions_from_txt, run_batch_queries, rows_to_csv_bytes
from rag_service import get_default_prompt_prefix, run_rag_query

from secda_index_utils import check_secda_index_ready

DEFAULT_PERSIST_DIR = "data/chroma_secda"
DEFAULT_COLLECTION_NAME = "secda_docs"
DEFAULT_RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="SECDA RAG Interface",
    page_icon="🧠",
    layout="wide",
)

st.title("SECDA RAG Interface")
st.caption(
    "Query SECDA-TFLite documentation using local Ollama models and a Chroma-backed retrieval pipeline."
)

MODE_DESCRIPTIONS = {
    "baseline": "No retrieval. The model answers without retrieved SECDA context.",
    "rag_strict": "Retrieves SECDA chunks and answers using only the retrieved SECDA context.",
    "rag_assisted": "Retrieves SECDA chunks and uses them as primary context.",
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
        value="data/chroma_secda",
    )

    collection_name = st.text_input(
        "Collection name",
        value="secda_docs_v1",
    )

    retriever_model = st.text_input(
        "Retriever model",
        value=DEFAULT_RETRIEVER_MODEL,
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

st.subheader("SECDA Index Status")

index_ready, index_message = check_secda_index_ready(
    persist_dir=persist_dir,
    collection_name=collection_name,
)

if index_ready:
    st.success("SECDA Index Ready")
else:
    st.warning(index_message)
    st.info(
        "To build the SECDA index, open the 'Process SECDA Documents' page from the Streamlit sidebar navigation."
    )

    if prompt_mode in RAG_MODES:
        st.error(
            "RAG modes require the SECDA documents to be processed first. "
            "Please build the SECDA index before using 'rag_strict' or 'rag_assisted'."
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

run_button = st.button("Run", type="primary")

if run_button:
    if not query.strip():
        st.warning("Please enter a prompt.")
    elif prompt_mode in RAG_MODES and not index_ready:
        st.error(
            "The SECDA index is not ready yet. "
            "Please open the 'Process SECDA Documents' page and build the index first."
        )
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
                st.write(f"**Retrieval latency:** {result.get('retrieval_latency_seconds', 0.0):.3f}s")
                st.write(f"**Generation latency:** {result.get('generation_latency_seconds', 0.0):.3f}s")
                st.write(f"**Total latency:** {result.get('total_latency_seconds', 0.0):.3f}s")

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

st.markdown("---")
st.subheader("Batch Evaluation")
st.caption(
    "Upload a .txt file with one question per line. Each question will be run in baseline, rag_strict, and rag_assisted modes."
)

uploaded_questions_file = st.file_uploader(
    "Upload question file (.txt)",
    type=["txt"],
    help="Use one question per line. Empty lines are ignored.",
)

if uploaded_questions_file is not None:
    try:
        uploaded_text = uploaded_questions_file.read().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The uploaded file could not be decoded as UTF-8 text.")
    else:
        batch_questions = parse_questions_from_txt(uploaded_text)

        if not batch_questions:
            st.warning("No questions were found. Please upload a .txt file with one question per line.")
        else:
            st.write(f"Parsed **{len(batch_questions)}** questions from **{uploaded_questions_file.name}**.")

            with st.expander("Preview parsed questions"):
                for i, batch_question in enumerate(batch_questions, start=1):
                    st.write(f"{i}. {batch_question}")

            run_batch_button = st.button("Run batch evaluation", type="primary")

            if run_batch_button:
                if not index_ready:
                    st.error(
                        "The SECDA index is not ready yet. "
                        "Please build the SECDA index before running batch evaluation."
                    )
                else:
                    batch_prompt_overrides = {
                        "baseline": get_default_prompt_prefix("baseline"),
                        "rag_strict": get_default_prompt_prefix("rag_strict"),
                        "rag_assisted": get_default_prompt_prefix("rag_assisted"),
                    }

                    with st.spinner("Running all uploaded questions across all three modes..."):
                        try:
                            batch_rows = run_batch_queries(
                                questions=batch_questions,
                                ollama_host=ollama_host,
                                ollama_model=selected_model,
                                n_results=n_results,
                                persist_dir=persist_dir,
                                collection_name=collection_name,
                                retriever_model=retriever_model,
                                max_new_tokens=max_new_tokens,
                                prompt_prefix_overrides=batch_prompt_overrides,
                            )
                        except Exception as exc:
                            st.error(str(exc))
                        else:
                            st.success(
                                f"Completed {len(batch_rows)} runs for {len(batch_questions)} questions across 3 modes."
                            )
                            st.dataframe(batch_rows, use_container_width=True)

                            if batch_rows:
                                avg_retrieval = sum(row.get("retrieval_latency_seconds", 0.0) for row in batch_rows) / len(batch_rows)
                                avg_generation = sum(row.get("generation_latency_seconds", 0.0) for row in batch_rows) / len(batch_rows)
                                avg_total = sum(row.get("total_latency_seconds", 0.0) for row in batch_rows) / len(batch_rows)

                                st.subheader("Batch Latency Summary")
                                st.write(f"**Average retrieval latency:** {avg_retrieval:.3f}s")
                                st.write(f"**Average generation latency:** {avg_generation:.3f}s")
                                st.write(f"**Average total latency:** {avg_total:.3f}s")

                            csv_bytes = rows_to_csv_bytes(batch_rows)
                            st.download_button(
                                label="Download results CSV",
                                data=csv_bytes,
                                file_name="batch_eval_results.csv",
                                mime="text/csv",
                            )