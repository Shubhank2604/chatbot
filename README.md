# Data Assistant

A local RAG data-discovery chatbot built with Streamlit, Chroma, LangChain, and Ollama. The app ingests `source_file.csv`, stores local embeddings in Chroma, retrieves relevant fields for a user question, and uses a local Llama model to generate an answer.

## What It Does

- Reads a structured CSV data dictionary from the project root
- Converts each CSV row into a LangChain `Document`
- Creates embeddings with Ollama's local `nomic-embed-text` model
- Stores embedded documents in a local Chroma vector database
- Runs a Streamlit chatbot UI
- Uses intent detection, query reformulation, query decomposition, vector retrieval, and grounded answer generation
- Optionally supports cross-encoder reranking

## Project Structure

- `app.py` - Streamlit chatbot application
- `ingest.py` - CSV ingestion and Chroma DB builder
- `run_sql.py` - Chroma inspection and reset utility
- `source_file.csv` - source data dictionary used for ingestion
- `chroma_db/` - persisted local Chroma database created after ingestion
- `.venv/` - local Python virtual environment
- `key.txt` - optional only if using a remote OpenAI-compatible API instead of Ollama

## Requirements

- Python 3.13 or compatible Python 3.x environment
- Ollama installed locally
- The following Ollama models:
  - `llama3.2`
  - `nomic-embed-text`

No paid API key is required for the default local Ollama setup.

## Install Python Dependencies

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -U streamlit pandas openai langchain-core langchain-openai langchain-chroma chromadb sentence-transformers
```

## Install Ollama Models

If `ollama` is on PATH:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

If PowerShell does not recognize `ollama`, use the full Windows path:

```powershell
& $env:LOCALAPPDATA\Programs\Ollama\ollama.exe pull llama3.2
& $env:LOCALAPPDATA\Programs\Ollama\ollama.exe pull nomic-embed-text
```

You can verify the local Ollama API with:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:11434/api/tags -UseBasicParsing
```

## Default Configuration

The app defaults to local Ollama:

```text
OPENAI_BASE_URL=http://localhost:11434/v1
CHAT_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
ENABLE_RERANKER=false
```

The code uses Ollama's OpenAI-compatible API. For local Ollama, the app automatically uses a dummy API key value, so `key.txt` is not needed.

To use a different local model:

```powershell
$env:CHAT_MODEL="llama3"
```

To enable the optional cross-encoder reranker:

```powershell
$env:ENABLE_RERANKER="true"
```

Reranking is disabled by default because first-time model loading can slow down or block a classroom demo. Vector retrieval still works without it.

## Ingest Data

Build the local Chroma database:

```powershell
.\.venv\Scripts\python.exe ingest.py
```

Expected output:

```text
Loading data from source_file.csv...
Processed 74 column definitions.
Initializing embedding client...
Creating/Updating Vector Database...
Success! Database created at ./chroma_db
```

Verify the document count:

```powershell
.\.venv\Scripts\python.exe run_sql.py count
```

Expected output:

```text
collection_count=74
```

## Run The App

Start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Do not run the app with `python app.py`; Streamlit apps need to be launched with `streamlit run`.

## Inspect Chroma

Useful commands:

```powershell
.\.venv\Scripts\python.exe run_sql.py count
.\.venv\Scripts\python.exe run_sql.py collections
.\.venv\Scripts\python.exe run_sql.py tables
.\.venv\Scripts\python.exe run_sql.py table-counts
.\.venv\Scripts\python.exe run_sql.py schema collections
.\.venv\Scripts\python.exe run_sql.py sample collections --limit 5
```

To delete the local Chroma database for a clean rebuild:

```powershell
.\.venv\Scripts\python.exe run_sql.py reset-db
```

## Current Retrieval Pipeline

1. User submits a question in the Streamlit chat UI.
2. The app classifies the question intent.
3. Follow-up questions are reformulated using chat history.
4. Broad questions are decomposed into focused subqueries.
5. Chroma performs maximum marginal relevance retrieval.
6. Optional reranking can reorder candidates if `ENABLE_RERANKER=true`.
7. Retrieved context is formatted and passed to the local Llama model.
8. The answer is streamed back into the UI.
9. A grounding check warns if the answer mentions field names not present in retrieved context.

## Known Limitations

- `source_file.csv` currently contains column names but blank definitions, examples, uses, notes, and aliases. The app can run, but answer quality will be limited until the CSV is enriched.
- Re-running ingestion without resetting `chroma_db` may duplicate records.
- Local Llama responses are free but may be slower and less capable than paid hosted models.
- The reranker is disabled by default for demo reliability.
- Chroma's SQLite files are implementation details and should not be edited manually.

## Demo Checklist

1. Confirm Ollama is running.
2. Confirm `llama3.2` and `nomic-embed-text` are installed.
3. Run ingestion.
4. Verify `collection_count=74`.
5. Start Streamlit.
6. Open `http://localhost:8501`.
7. Ask about a field present in `source_file.csv`, such as `UF_GPA`, `TERM_CD`, or `FIRST_GEN_IND`.
