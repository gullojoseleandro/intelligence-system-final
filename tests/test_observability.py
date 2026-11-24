import os
import unittest
from unittest.mock import patch

from app.observability import load_settings


class ObservabilitySettingsTests(unittest.TestCase):
    def test_live_mode_requires_model_key(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODE": "openai", "OPENAI_API_KEY": "", "LANGSMITH_TRACING": "false"}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                load_settings()

    def test_tracing_requires_langsmith_key(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODE": "offline", "LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "LANGSMITH_API_KEY"):
                load_settings()

    def test_nvidia_mode_requires_its_own_key(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODE": "nvidia", "NVIDIA_API_KEY": "", "LANGSMITH_TRACING": "false"}):
            with self.assertRaisesRegex(ValueError, "NVIDIA_API_KEY"):
                load_settings()

    def test_nvidia_mode_accepts_a_configured_key(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODE": "nvidia", "NVIDIA_API_KEY": "nvapi-test", "LANGSMITH_TRACING": "false"}):
            mode, _ = load_settings()
        self.assertEqual(mode, "nvidia")

    def test_ollama_and_phoenix_need_no_cloud_keys(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODE": "ollama", "OPENAI_API_KEY": "", "NVIDIA_API_KEY": "", "LANGSMITH_TRACING": "false", "PHOENIX_TRACING": "true", "LANGSMITH_API_KEY": ""}):
            mode, _ = load_settings()
        self.assertEqual(mode, "ollama")
