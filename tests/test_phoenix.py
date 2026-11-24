import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.observability import configure_phoenix_tracing


ROOT = Path(__file__).resolve().parents[1]


class PhoenixTracingTests(unittest.TestCase):
    def test_enabled_tracing_instruments_langchain_with_local_exporter(self) -> None:
        with patch.dict(os.environ, {"PHOENIX_TRACING": "true", "PHOENIX_COLLECTOR_ENDPOINT": "http://phoenix:6006/v1/traces", "LANGSMITH_API_KEY": ""}), \
             patch("app.observability.LangChainInstrumentor") as instrumentor, \
             patch("app.observability.trace") as trace, \
             patch("app.observability.OTLPSpanExporter") as exporter, \
             patch("app.observability.TracerProvider") as provider, \
             patch("app.observability.BatchSpanProcessor") as processor:
            tracer = configure_phoenix_tracing()
        exporter.assert_called_once_with(endpoint="http://phoenix:6006/v1/traces")
        processor.assert_called_once_with(exporter.return_value)
        provider.return_value.add_span_processor.assert_called_once_with(processor.return_value)
        trace.set_tracer_provider.assert_called_once_with(provider.return_value)
        instrumentor.return_value.instrument.assert_called_once_with(tracer_provider=provider.return_value)
        self.assertIs(tracer, provider.return_value.get_tracer.return_value)

    def test_disabled_tracing_does_not_create_exporter(self) -> None:
        with patch.dict(os.environ, {"PHOENIX_TRACING": "false"}), patch("app.observability.OTLPSpanExporter") as exporter:
            self.assertIsNone(configure_phoenix_tracing())
        exporter.assert_not_called()

    def test_compose_exposes_local_persistent_phoenix(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("phoenix:", compose)
        self.assertIn("127.0.0.1:6006:6006", compose)
        self.assertIn("phoenix-data:", compose)
        self.assertIn("PHOENIX_COLLECTOR_ENDPOINT", compose)
        self.assertIn('PHOENIX_ENABLE_AUTH: "false"', compose)
