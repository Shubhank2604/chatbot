import argparse
import shutil
import sqlite3
from pathlib import Path

from langchain_chroma import Chroma

DB_DIR = Path("./chroma_db")
SQLITE_FILE = DB_DIR / "chroma.sqlite3"


def _connect() -> sqlite3.Connection:
    if not SQLITE_FILE.exists():
        raise FileNotFoundError(f"Chroma SQLite file not found: {SQLITE_FILE}")
    return sqlite3.connect(SQLITE_FILE)


def show_collection_count() -> None:
    db = Chroma(persist_directory=str(DB_DIR))
    print(f"collection_count={db._collection.count()}")


def list_collections() -> None:
    with _connect() as conn:
        rows = conn.execute("select id, name from collections order by name").fetchall()
    if not rows:
        print("No collections found.")
        return
    for collection_id, name in rows:
        print(f"{name} | {collection_id}")


def list_tables() -> None:
    with _connect() as conn:
        rows = conn.execute(
            "select name from sqlite_master where type='table' order by name"
        ).fetchall()
    for (name,) in rows:
        print(name)


def table_counts() -> None:
    with _connect() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type='table' order by name"
            ).fetchall()
        ]
        for table in tables:
            count = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
            print(f"{table}: {count}")


def show_schema(table: str) -> None:
    with _connect() as conn:
        row = conn.execute(
            "select sql from sqlite_master where type='table' and name = ?",
            (table,),
        ).fetchone()
    if not row or not row[0]:
        raise ValueError(f"Table not found: {table}")
    print(row[0])


def sample_rows(table: str, limit: int) -> None:
    with _connect() as conn:
        cursor = conn.execute(f'select * from "{table}" limit {limit}')
        columns = [desc[0] for desc in cursor.description]
        print(" | ".join(columns))
        for row in cursor.fetchall():
            print(" | ".join("" if value is None else str(value) for value in row))


def run_query(query: str) -> None:
    with _connect() as conn:
        cursor = conn.execute(query)
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            print(" | ".join(columns))
            for row in cursor.fetchall():
                print(" | ".join("" if value is None else str(value) for value in row))
        else:
            conn.commit()
            print("Query executed.")


def reset_db() -> None:
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print(f"Deleted {DB_DIR}")
    else:
        print(f"{DB_DIR} does not exist.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and manage the local Chroma DB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("count", help="Show Chroma collection document count")
    subparsers.add_parser("collections", help="List Chroma collections")
    subparsers.add_parser("tables", help="List SQLite table names")
    subparsers.add_parser("table-counts", help="Show row counts for all SQLite tables")

    schema_parser = subparsers.add_parser("schema", help="Show CREATE TABLE SQL")
    schema_parser.add_argument("table", help="SQLite table name")

    sample_parser = subparsers.add_parser("sample", help="Show sample rows from a table")
    sample_parser.add_argument("table", help="SQLite table name")
    sample_parser.add_argument("--limit", type=int, default=5, help="Row limit")

    query_parser = subparsers.add_parser("query", help="Run a raw SQL query")
    query_parser.add_argument("sql", help="SQL to execute against chroma.sqlite3")

    subparsers.add_parser("reset-db", help="Delete the entire local chroma_db folder")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "count":
        show_collection_count()
    elif args.command == "collections":
        list_collections()
    elif args.command == "tables":
        list_tables()
    elif args.command == "table-counts":
        table_counts()
    elif args.command == "schema":
        show_schema(args.table)
    elif args.command == "sample":
        sample_rows(args.table, args.limit)
    elif args.command == "query":
        run_query(args.sql)
    elif args.command == "reset-db":
        reset_db()


if __name__ == "__main__":
    main()
