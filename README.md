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

## Requirements

- Python 3.13 or compatible Python 3.x environment
- Ollama installed locally
- Ollama models:
  - `llama3.2`
  - `nomic-embed-text`
- Python dependencies installed in `.venv`
- No paid API key is required for the default local Ollama setup

## Architecture

### System Flow

```text
source_file.csv
      |
      v
ingest.py
      |
      v
LangChain Documents
      |
      v
Ollama Embeddings
nomic-embed-text
      |
      v
Chroma Vector DB
./chroma_db
      |
      v
app.py Streamlit UI
      |
      v
User question -> retrieval -> local Llama answer
```

### Ingestion Flow

```text
CSV row
  |
  |  COLUMN_NAME
  |  DEFINITION
  |  COMMON_USES
  |  EXAMPLES
  |  GRAIN_SCOPE
  |  NOTES
  |  ALIAS_KEYWORDS
  v
LangChain Document
  |
  | page_content = labeled field description
  | metadata = column_name, source, doc_type
  v
Embedding request to Ollama
  |
  v
Vector stored in Chroma
```

`ingest.py` is responsible for preparing the retrieval database. It reads `source_file.csv`, creates one document per column definition, sends those documents to Ollama's embedding endpoint, and persists the resulting vectors in Chroma.

### Chat Flow

```text
User asks a question
      |
      v
Intent detection
      |
      v
Question reformulation
      |
      v
Query decomposition
      |
      v
Chroma MMR retrieval
      |
      v
Optional reranking
      |
      v
Context formatting
      |
      v
Llama response generation
      |
      v
Grounding check + Streamlit answer
```

`app.py` is responsible for the interactive RAG experience. It classifies the user's intent, rewrites follow-up questions when needed, retrieves relevant data dictionary entries from Chroma, formats the retrieved context, and asks the local Llama model to answer using that context.

### Runtime Components

```text
Streamlit
  - Chat UI
  - Sidebar settings
  - Chat history

LangChain
  - Prompt templates
  - Message history
  - Chat model wrapper
  - Embedding wrapper

Ollama
  - llama3.2 for generation
  - nomic-embed-text for embeddings
  - OpenAI-compatible local API at http://localhost:11434/v1

Chroma
  - Local vector store
  - Persists under ./chroma_db
  - Supports semantic retrieval
```

### Important Limitations

- `source_file.csv` currently has column names but mostly blank descriptions, examples, notes, and aliases. The app runs, but answer quality depends heavily on enriching this CSV.
- Re-running ingestion without resetting `chroma_db` may duplicate records.
- Local Llama is free but less capable than large hosted models.
- The cross-encoder reranker is disabled by default for demo reliability.

## Run The Project

```powershell
.\.venv\Scripts\python.exe -m pip install -U streamlit pandas openai langchain-core langchain-openai langchain-chroma chromadb sentence-transformers
ollama pull llama3.2
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe run_sql.py count
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the app at:

```text
http://localhost:8501
```
