import tempfile
import unittest
import gc
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.embeddings import Embeddings
from chromadb.api.client import SharedSystemClient

from app.agents.research_agent import research_node
from app.rag.ingest import ingest_documents
from app.rag.retrieval import retrieve_reports


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.documents_embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower()
        return [float(words.count("cache")), float(words.count("queue")), 1.0]


class AsyncOnlyEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("synchronous document embeddings called")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("synchronous query embedding called")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [FakeEmbeddings.embed_query(self, text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return FakeEmbeddings.embed_query(self, text)


class RagTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingestion_and_retrieval_use_native_async_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "INC-300.md").write_text("cache failure", encoding="utf-8")
            embeddings = AsyncOnlyEmbeddings()
            try:
                self.assertEqual(await ingest_documents(data, root / "vectors", embeddings, "async-v1"), 1)
                reports = await retrieve_reports("cache", root / "vectors", embeddings, "async-v1")
                self.assertEqual(reports[0]["source_id"], "INC-300")
            finally:
                SharedSystemClient.clear_system_cache()
                gc.collect()

    async def test_ingestion_is_persistent_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "INC-300.md").write_text("cache " * 600, encoding="utf-8")
            embeddings = FakeEmbeddings()

            first = await ingest_documents(data, root / "vectors", embeddings, "fake-v1")
            second = await ingest_documents(data, root / "vectors", embeddings, "fake-v1")
            reports = await retrieve_reports("cache", root / "vectors", embeddings, "fake-v1")

            self.assertGreater(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(embeddings.documents_embedded, first)
            self.assertEqual(reports[0]["source_id"], "INC-300")
            self.assertEqual(reports[0]["path"], "data/INC-300.md")
            self.assertIn("cache", reports[0]["text"])
            SharedSystemClient.clear_system_cache()
            gc.collect()

    async def test_embedding_model_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "INC-300.md").write_text("cache failure", encoding="utf-8")
            await ingest_documents(data, root / "vectors", FakeEmbeddings(), "fake-v1")

            with self.assertRaisesRegex(ValueError, "embedding model"):
                await retrieve_reports("cache", root / "vectors", FakeEmbeddings(), "fake-v2")
            SharedSystemClient.clear_system_cache()
            gc.collect()

    async def test_research_restricts_explicit_incident_and_deduplicates(self) -> None:
        dense = [
            {"source_id": "INC-207", "path": "data/INC-207.md", "text": "dense duplicate"},
            {"source_id": "INC-300", "path": "data/INC-300.md", "text": "dense result"},
        ]
        with patch("app.agents.research_agent.retrieve_reports", new_callable=AsyncMock, return_value=dense):
            result = await research_node({"query": "INC-207 queue"})

        self.assertEqual([report["source_id"] for report in result["reports"]], ["INC-207"])
        self.assertNotEqual(result["reports"][0]["text"], "dense duplicate")
