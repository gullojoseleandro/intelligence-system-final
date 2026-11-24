"""Idempotently index local text and Markdown reports in Chroma."""

import argparse
import asyncio
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TokenTextSplitter

from app.rag.store import ollama_embeddings, open_store, settings


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _documents(data_dir: Path) -> dict[str, Document]:
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    documents: dict[str, Document] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        relative = path.relative_to(data_dir).as_posix()
        source_path = f"data/{relative}"
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(splitter.split_text(text)):
            if not chunk.strip():
                continue
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
            key = hashlib.sha256(f"{source_path}:{index}:{digest}".encode()).hexdigest()
            documents[key] = Document(
                page_content=chunk,
                metadata={"source_id": path.stem.upper(), "path": source_path, "chunk": index},
            )
    return documents


async def ingest_documents(
    data_dir: Path,
    db_path: Path,
    embeddings: Embeddings,
    embed_model: str,
) -> int:
    """Add only changed chunks and remove stale chunks from the configured corpus."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {data_dir}")
    documents = await asyncio.to_thread(_documents, data_dir)
    store = await open_store(db_path, embeddings, embed_model)
    existing = await asyncio.to_thread(store.get, include=["metadatas"])
    existing_ids = set(existing["ids"])
    stale = list(existing_ids - documents.keys())
    if stale:
        await asyncio.to_thread(store.delete, stale)
    fresh = list(documents.keys() - existing_ids)
    if fresh:
        vectors = await embeddings.aembed_documents([documents[key].page_content for key in fresh])
        await asyncio.to_thread(
            store._collection.upsert,
            ids=fresh,
            embeddings=vectors,
            documents=[documents[key].page_content for key in fresh],
            metadatas=[documents[key].metadata for key in fresh],
        )
    return len(fresh)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index local Markdown and text reports in Chroma")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    db_path, model = settings()
    count = await ingest_documents(args.data_dir, db_path, ollama_embeddings(model), model)
    print(f"Indexed {count} new or changed chunk(s).")


if __name__ == "__main__":
    asyncio.run(main())
