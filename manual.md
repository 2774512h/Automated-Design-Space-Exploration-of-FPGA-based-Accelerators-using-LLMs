# User manual

This manual describes how to use the Streamlit RAG application for SECDA-TFLite.

## 1. Purpose

The application lets users process a local SECDA-TFLite codebase into a vector index and then query that indexed content through a local Ollama-backed RAG interface.

The system has two parts:

1. a preprocessing and indexing workflow
2. a querying and evaluation workflow

## 2. Before you start

Open a terminal in the project root directory:
`Automated-Design-Space-Exploration-of-FPGA-based-Accelerators-using-LLMs`

Create a virtual environment:

```bash
python -m venv .venv

Activate it on Windows:
.\.venv\Scripts\activate

You will need:

- a working Python environment
- the packages listed in `requirements.txt`
- the spaCy English model `en_core_web_sm`
- a local Ollama installation
- at least one installed Ollama model
- a local SECDA-TFLite repository on your machine

Recommended startup command:

```bash
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

## 3. Starting the application

Run:

```bash
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

This opens the Streamlit app in your browser.

The application contains:

- a main RAG interface page
- a **Process SECDA Documents** page

## 4. First-time setup: build the SECDA index

Before using RAG modes, you need to build the SECDA document index.

### Step 1: Open the processing page

In the Streamlit sidebar, open:

**Process SECDA Documents**

### Step 2: Check the current index status

The page displays the current SECDA index status. If no index exists yet, a warning will be shown.

### Step 3: Enter the SECDA-TFLite repository path

In the field for the SECDA-TFLite root directory, enter the full local path to the SECDA-TFLite repository.

This must be a local directory on the machine running the app.

### Step 4: Run preprocessing or full build

You can use:

- **Run Preprocess**
- **Build Chroma Index**
- **Run Full SECDA Build**

For first-time setup, use:

**Run Full SECDA Build**

This runs:

1. `src/preprocess.py`
2. `src/index_secda_jsonl.py`

The preprocessing script scans the SECDA-TFLite directory for `.md`, `.rst`, and `.txt` files, splits them into chunks, and writes them to `data/processed/secda_docs_chunks.jsonl`.

The indexing script reads that JSONL file and builds a Chroma vector collection.

### Step 5: Confirm successful build

A successful run should show:

- preprocessing completed successfully
- Chroma index build completed successfully
- SECDA index is ready

The standard build target is:

- persist directory: `data/chroma_secda`
- collection name: `secda_docs_v1`

## 5. Using the main RAG interface

Open the main Streamlit page.

### Step 1: Set the configuration

In the sidebar, configure:

- **Ollama host**
- **Chroma persist directory**
- **Collection name**
- **Retriever model**
- **Prompt mode**
- **Top-k retrieved chunks** for RAG modes
- **Max new tokens**

The app supports three prompt modes:

- `baseline`
- `rag_strict`
- `rag_assisted`

### Step 2: Use the correct SECDA index settings

Set these values to match the built SECDA index:

- `data/chroma_secda`
- `secda_docs_v1`

### Step 3: Select an Ollama model

The app checks whether Ollama is running and lists installed local models before allowing inference.

Choose one installed model from the dropdown.

### Step 4: Enter your prompt

Type your question into the prompt box.

Example query types:

- questions about a specific SECDA-TFLite file
- questions about how a configuration file is used
- questions about the purpose of a module or document
- comparison queries across indexed documentation

### Step 5: Run the query

Click **Run**.

The app will:

- build the final prompt
- retrieve SECDA document chunks if using a RAG mode
- send the prompt to the selected Ollama model
- display the answer and run summary

If using `rag_strict` or `rag_assisted`, the app can also display the retrieved context used during answering.

## 6. Batch evaluation

The main page also supports batch evaluation.

### Step 1: Upload a question file

Upload a `.txt` file with one question per line.

### Step 2: Run evaluation

Click **Run batch evaluation**.

The app will run all questions across:

- `baseline`
- `rag_strict`
- `rag_assisted`

and display the results in a table. It also allows downloading a CSV of the results.

## 7. Troubleshooting

### Problem: SECDA index is not ready

Cause:
- preprocessing has not been run
- indexing has not completed
- the wrong persist directory or collection name is being used

Fix:
- run **Run Full SECDA Build**
- confirm the main page is using `data/chroma_secda` and `secda_docs_v1`

### Problem: spaCy model not found

Cause:
- `en_core_web_sm` is not installed

Fix:
```bash
python -m spacy download en_core_web_sm
```

### Problem: `spacy_chunks` factory not found

Cause:
- the environment does not contain the `spacy-chunks` package
- or the app is being run in a different Python environment from the one where it is installed

Fix:
- install the package listed in `requirements.txt`
- make sure Streamlit is launched from the correct Python environment

### Problem: Ollama models are not available

Cause:
- Ollama is not running
- or no models are installed

Fix:
- start Ollama
- confirm at least one local model is installed
- re-open the app or refresh the page

### Problem: RAG answer is too generic

Cause:
- wrong index settings
- baseline mode selected
- insufficient relevant indexed documentation

Fix:
- verify `rag_strict` or `rag_assisted` is selected
- verify index settings match the SECDA collection
- inspect retrieved context in the expander to confirm relevant chunks were returned

## 8. Expected outputs

When everything is working correctly, users should be able to:

- process a local SECDA-TFLite repository into document chunks
- build a Chroma index from those chunks
- query the indexed SECDA content through the Streamlit interface
- compare baseline and RAG outputs
- run multi-question batch evaluation

## 9. Summary of workflow

1. Install dependencies
2. Install the spaCy English model
3. Start the Streamlit app
4. Build the SECDA index from the processing page
5. Open the main RAG page
6. Configure the SECDA index values
7. Choose an Ollama model
8. Ask questions or run batch evaluation