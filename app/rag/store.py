"""Shared Chroma configuration and validated report contract."""

import asyncio
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, ConfigDict, Field


COLLECTION = "research_reports"


class ReportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    text: str = Field(min_length=1)


def settings() -> tuple[Path, str]:
    return Path(os.getenv("RAG_DB_PATH", "vectorstore")), os.getenv(
        "OLLAMA_EMBED_MODEL", "embeddinggemma:300m-qat-q4_0"
    )


def ollama_embeddings(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _open_store(path: Path, embeddings: Embeddings, model: str) -> Chroma:
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(path),
        embedding_function=embeddings,
        collection_metadata={"embedding_model": model},
    )
    actual = (store._collection.metadata or {}).get("embedding_model")
    if actual != model:
        raise ValueError(f"Chroma embedding model mismatch: expected {model!r}, found {actual!r}")
    return store


async def open_store(path: Path, embeddings: Embeddings, model: str) -> Chroma:
    # Chroma's constructor opens a local SQLite database synchronously.
    return await asyncio.to_thread(_open_store, path, embeddings, model)
