import os
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from app.supervisor import NvidiaSupervisor, OpenAISupervisor, OllamaSupervisor, RouteDecision, supervisor_node
from app.graph import initial_state


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

    def test_ollama_uses_local_structured_output(self) -> None:
        with patch("app.supervisor.ChatOllama") as model_class:
            model_class.return_value.with_structured_output.return_value = RunnableLambda(
                lambda _: RouteDecision(next_agent="research", reason="need sources")
            )
            OllamaSupervisor("gemma3:1b", base_url="http://ollama:11434")
        model_class.assert_called_once_with(model="gemma3:1b", base_url="http://ollama:11434", temperature=0)
        model_class.return_value.with_structured_output.assert_called_once_with(RouteDecision, method="json_schema")


class SupervisorProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_evidence_advances_to_analysis_even_if_router_repeats_research(self) -> None:
        class RepeatingRouter:
            async def decide(self, state):
                return RouteDecision(next_agent="research", reason="retry research")

        state = initial_state("Investigate INC-421")
        state["reports"] = [{"source_id": "INC-421", "path": "data/INC-421.md", "text": "baseline_p95_ms: 170\ncurrent_p95_ms: 260"}]
        state["supervisor_steps"] = 1
        result = await supervisor_node(state, RepeatingRouter())
        self.assertEqual(result["next_agent"], "analysis")
