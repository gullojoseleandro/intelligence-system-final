"""Async semantic retrieval with validated report output."""

import asyncio
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.rag.store import ReportRecord, ollama_embeddings, open_store, settings
from app.state import ResearchReport


async def retrieve_reports(
    query: str,
    db_path: Path | None = None,
    embeddings: Embeddings | None = None,
    embed_model: str | None = None,
) -> list[ResearchReport]:
    """Retrieve citable chunks without blocking the event loop on local I/O."""
    configured_path, configured_model = settings()
    path = db_path if db_path is not None else configured_path
    model = embed_model if embed_model is not None else configured_model
    if not path.exists() or not query.strip():
        return []
    embedder = embeddings or ollama_embeddings(model)
    store = await open_store(path, embedder, model)
    vector = await embedder.aembed_query(query)
    matches = await asyncio.to_thread(
        store._collection.query,
        query_embeddings=[vector],
        n_results=3,
        include=["documents", "metadatas"],
    )
    reports: list[ResearchReport] = []
    seen: set[str] = set()
    for source_id, path, text in zip(
        (metadata.get("source_id") for metadata in matches["metadatas"][0]),
        (metadata.get("path") for metadata in matches["metadatas"][0]),
        matches["documents"][0],
    ):
        report = ReportRecord.model_validate(
            {
                "source_id": source_id,
                "path": path,
                "text": text,
            }
        ).model_dump()
        if report["source_id"] not in seen:
            reports.append(report)
            seen.add(report["source_id"])
    return reports
