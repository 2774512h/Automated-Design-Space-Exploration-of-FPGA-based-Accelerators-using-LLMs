# RAG APPLICATION FOR SECDA-TFLITE

This project is a Streamlit-based Retrieval-Augmented Generation (RAG) application for the SECDA-TFLite codebase.

Its purpose is to:
- create a vectorised version of the SECDA-TFLite codebase
- allow users to experiment with and evaluate the RAG model
- support future use of a local fine-tuned model through Ollama to produce FPGA-based accelerator designs

This README covers the Streamlit RAG application only.

---

## Project structure

```text
Automated-Design-Space-Exploration-of-FPGA-based-Accelerators-using-LLMs/
├── .venv/
├── app/
│   ├── __pycache__/
│   ├── pages/
│   │   └── 1_Process_SECDA_Documents.py
│   └── streamlit_app.py
├── data/
│   ├── chroma/
│   ├── chroma_secda/
│   ├── index/
│   ├── processed/
│   └── raw/
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── add_groups.py
│   ├── batch_eval.py
│   ├── chroma_index.py
│   ├── experiment_infer.py
│   ├── index_secda_jsonl.py
│   ├── inspect_collection.py
│   ├── llm.py
│   ├── ollama_client.py
│   ├── preprocess.py
│   ├── query_rag.py
│   ├── query_tree_rag.py
│   ├── rag_service.py
│   ├── secda_index_utils.py
│   ├── tree_retrieval.py
│   └── vectorise.py
├── .gitignore
├── LICENSE
├── ReadMe.md
└── requirements.txt