import os
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from app.supervisor import NvidiaSupervisor, OpenAISupervisor, RouteDecision


class SupervisorProviderTests(unittest.TestCase):
    def test_nvidia_uses_its_endpoint_and_function_calling(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}):
            with patch("app.supervisor.ChatOpenAI") as model_class:
                model_class.return_value.with_structured_output.return_value = RunnableLambda(
                    lambda _: RouteDecision(next_agent="research", reason="need sources")
                )
                router = NvidiaSupervisor()

        self.assertIsNotNone(router.chain)
        self.assertEqual(model_class.call_args.kwargs["api_key"], "nvapi-test")
        self.assertEqual(model_class.call_args.kwargs["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(model_class.call_args.kwargs["model"], "meta/llama-3.1-8b-instruct")
        model_class.return_value.with_structured_output.assert_called_once_with(
            RouteDecision, method="function_calling"
        )

    def test_openai_keeps_existing_model_configuration(self) -> None:
        with patch("app.supervisor.ChatOpenAI") as model_class:
            model_class.return_value.with_structured_output.return_value = RunnableLambda(
                lambda _: RouteDecision(next_agent="research", reason="need sources")
            )
            OpenAISupervisor("gpt-4o-mini")

        model_class.assert_called_once_with(model="gpt-4o-mini", temperature=0)
        model_class.return_value.with_structured_output.assert_called_once_with(RouteDecision)
