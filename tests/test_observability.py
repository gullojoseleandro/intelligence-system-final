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
