import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI

# --- Configuration ---
DATA_FILE = "source_file.csv"
DB_PATH = "./chroma_db"
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def load_api_key(path: str = "key.txt") -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key

    if BASE_URL.startswith("http://localhost:11434") or BASE_URL.startswith(
        "http://127.0.0.1:11434"
    ):
        return "ollama"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "No API key found. Set OPENAI_API_KEY or create key.txt in the project folder."
        )

    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError("No API key found. Set OPENAI_API_KEY or fill key.txt.")
    return key


class OpenAIEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for OpenAI or an OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str | None, model: str):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_and_process_data():
    print(f"Loading data from {DATA_FILE}...")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = {"COLUMN_NAME", "DEFINITION"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}. " f"Found: {list(df.columns)}"
        )

    documents = []
    for _, row in df.iterrows():
        col = _safe_str(row.get("COLUMN_NAME"))
        if not col:
            continue

        definition = _safe_str(row.get("DEFINITION"))
        uses = _safe_str(row.get("COMMON_USES"))
        examples = _safe_str(row.get("EXAMPLES"))
        grain = _safe_str(row.get("GRAIN_SCOPE"))
        notes = _safe_str(row.get("NOTES"))
        alias = _safe_str(row.get("ALIAS_KEYWORDS"))

        # RAG-optimized chunk: stable labels + user-intent phrasing
        content = "\n".join(
            [
                f"COLUMN_NAME: {col}",
                f"DEFINITION: {definition}",
                f"COMMON_USES: {uses}",
                f"EXAMPLES: {examples}",
                f"GRAIN_SCOPE: {grain}",
                f"NOTES: {notes}",
                f"ALIAS_KEYWORDS: {alias}",
                "CONTEXT: This entry describes a field available in the institutional data environment for discovery purposes.",
            ]
        ).strip()

        metadata = {
            "column_name": col,
            "source": "RAG_Data_Dictionary",
            "doc_type": "column_definition",
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Processed {len(documents)} column definitions.")
    return documents


def create_vector_db(documents):
    print("Initializing embedding client...")
    api_key = load_api_key("key.txt")

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=BASE_URL,
        model=EMBEDDING_MODEL,
    )

    print("Creating/Updating Vector Database...")

    # Optional: wipe existing DB for clean rebuild
    # (recommended when changing schemas/chunk format)
    # import shutil
    # if os.path.exists(DB_PATH):
    #     shutil.rmtree(DB_PATH)

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    print("Success! Database created at", DB_PATH)
    return vector_db


if __name__ == "__main__":
    docs = load_and_process_data()
    create_vector_db(docs)
