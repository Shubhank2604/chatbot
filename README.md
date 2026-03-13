# Data Discovery Assistant

An AI-powered data discovery assistant that ingests a curated data dictionary into a Chroma vector database, generates embeddings, and uses retrieval-augmented generation (RAG) to answer questions about available fields, definitions, feasibility, and data request scoping.

## What it does

- Loads a structured CSV data dictionary from the project root
- Converts each row into a searchable document with metadata
- Creates embeddings through an OpenAI-compatible API
- Stores the embedded documents in a local Chroma database
- Runs a Streamlit chat app for interactive question answering
- Uses retrieval, reranking, and an LLM to return grounded responses

## Project structure

- [app.py](c:\Users\shubhank.chandak\chatbot-backend\app.py): Streamlit chat application
- [ingest.py](c:\Users\shubhank.chandak\chatbot-backend\ingest.py): data ingestion and Chroma DB builder
- [run_sql.py](c:\Users\shubhank.chandak\chatbot-backend\run_sql.py): Chroma inspection and reset utility
- `key.txt`: API key file used for embeddings and chat model access
- `chroma_db/`: persisted local Chroma database
- `*.csv`: source data dictionary used for ingestion

## How it works

1. A CSV file in the root folder contains field-level metadata such as column name, definition, common uses, examples, notes, and aliases.
2. [ingest.py](c:\Users\shubhank.chandak\chatbot-backend\ingest.py) reads that file, converts each row into a LangChain `Document`, and writes the embeddings into `./chroma_db`.
3. [app.py](c:\Users\shubhank.chandak\chatbot-backend\app.py) loads the vector store, reranker, and chat model.
4. When a user asks a question, the app classifies intent, retrieves relevant documents from Chroma, reranks them, and generates a grounded answer in the chat UI.

## Requirements

- Python 3.13 or compatible Python 3.x environment
- A valid API key for the configured OpenAI-compatible endpoint
- Network access for:
  - embedding generation
  - chat model calls
  - first-time model download for the reranker

## Install dependencies

```powershell
py -m pip install streamlit langchain-core langchain-openai langchain-chroma sentence-transformers openai chromadb pandas
```

## Configuration

Create a `key.txt` file in the project root and paste only the raw API key value into it.

Expected layout:

```text
project-root/
  app.py
  ingest.py
  run_sql.py
  key.txt
  your_data_dictionary.csv
  chroma_db/
```

## Ingest data

Before running the app, build the local vector database:

```powershell
py ingest.py
```

Expected output:

```text
Loading data from ...
Processed <n> column definitions.
Initializing embedding client...
Creating/Updating Vector Database...
Success! Database created at ./chroma_db
```

## Run the app

Start the Streamlit UI with:

```powershell
py -m streamlit run app.py
```

Do not run the app with `py app.py`. Streamlit apps need to be launched with `streamlit run` so session state and chat UI features work correctly.

## Inspect the local Chroma DB

Use [run_sql.py](c:\Users\shubhank.chandak\chatbot-backend\run_sql.py) to inspect the database:

```powershell
py run_sql.py count
py run_sql.py collections
py run_sql.py tables
py run_sql.py table-counts
py run_sql.py schema collections
py run_sql.py sample collections --limit 5
py run_sql.py query "select id, name from collections"
py run_sql.py reset-db
```

Notes:

- `count` shows the Chroma collection document count
- `collections` lists available Chroma collections
- `reset-db` deletes the local `chroma_db` folder for a clean rebuild
- direct SQL edits are not recommended because Chroma manages its own internal schema

## Current retrieval pipeline

The chat application currently includes:

- intent detection for different query types
- follow-up question reformulation
- query decomposition for broader recall
- vector retrieval with maximum marginal relevance (MMR)
- cross-encoder reranking
- grounded answer generation
- optional source display
- lightweight confidence signaling

## Common workflow

1. Place the source CSV file in the project root.
2. Add the API key to `key.txt`.
3. Run `py ingest.py` to populate `chroma_db`.
4. Run `py -m streamlit run app.py`.
5. Ask questions in the browser UI.

## Known limitations

- Re-running ingestion without clearing the DB may duplicate records, depending on how documents are added.
- The app depends on external model and embedding APIs.
- The SQLite files inside `chroma_db` are implementation details of Chroma and should not be edited manually.

