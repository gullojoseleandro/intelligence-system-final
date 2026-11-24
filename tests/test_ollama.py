import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LocalStackTests(unittest.TestCase):
    def test_compose_contains_local_model_bootstrap_and_ingestion(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        for value in ("ollama:", "model-pull:", "ingest:", "python", "app.rag.ingest", "gemma3:1b", "embeddinggemma:300m-qat-q4_0", "service_completed_successfully", "ollama-data:", "vectorstore-data:"):
            self.assertIn(value, compose)

    def test_container_copies_corpus_and_app(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("COPY app ./app", dockerfile)
        self.assertIn("COPY data ./data", dockerfile)
