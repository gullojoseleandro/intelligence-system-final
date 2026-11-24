import json
import os
from typing import Literal, Protocol

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:  # The cloud-only test environment may not install local model extras.
    ChatOllama = None
from pydantic import BaseModel, Field
from langsmith import traceable

from app.state import OrchestratorState


MAX_SUPERVISOR_STEPS = 5


class RouteDecision(BaseModel):
    next_agent: Literal["research", "analysis", "finish"]
    reason: str = Field(min_length=1)


class Router(Protocol):
    async def decide(self, state: OrchestratorState) -> RouteDecision: ...


class OpenAISupervisor:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        structured_method: str | None = None,
    ) -> None:
        model_options = {"model": model_name, "temperature": 0}
        if api_key is not None:
            model_options["api_key"] = api_key
        if base_url is not None:
            model_options["base_url"] = base_url
        model = ChatOpenAI(**model_options)
        structured_options = {"method": structured_method} if structured_method else {}
        self.chain = ChatPromptTemplate.from_messages(
            [
                ("system", "You supervise two specialists. Choose research for missing or conflicting sources; choose analysis only after citable measurements exist; choose finish only when a source-backed calculation exists. Never invent evidence. A bounded graph guard may override your choice."),
                ("human", "Question: {query}\nResearch: {reports}\nAnalysis: {analysis}\nContributions: {contributions}\nSupervisor steps: {steps}"),
            ]
        ) | model.with_structured_output(RouteDecision, **structured_options)

    @traceable(name="supervisor_decision")
    async def decide(self, state: OrchestratorState) -> RouteDecision:
        return await self.chain.ainvoke(
            {
                "query": state["query"],
                "reports": json.dumps([{"source_id": report["source_id"], "text": report["text"][:1200]} for report in state["reports"]]),
                "analysis": json.dumps(state.get("analysis")),
                "contributions": json.dumps(state["contributions"][-4:]),
                "steps": state["supervisor_steps"],
            }
        )


class NvidiaSupervisor(OpenAISupervisor):
    """Route decisions through NVIDIA's hosted OpenAI-compatible NIM endpoint."""

    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct") -> None:
        super().__init__(
            model_name,
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
            structured_method="function_calling",
        )


class OllamaSupervisor(OpenAISupervisor):
    """Use a local Ollama model while retaining the validated routing contract."""

    def __init__(self, model_name: str = "gemma3:1b", *, base_url: str = "http://localhost:11434") -> None:
        if ChatOllama is None:
            raise RuntimeError("Install langchain-ollama before using MODEL_MODE=ollama")
        model = ChatOllama(model=model_name, base_url=base_url, temperature=0)
        self.chain = ChatPromptTemplate.from_messages(
            [
                ("system", "You supervise two specialists. Choose research for missing or conflicting sources; choose analysis only after citable measurements exist; choose finish only when a source-backed calculation exists. Never invent evidence. A bounded graph guard may override your choice."),
                ("human", "Question: {query}\nResearch: {reports}\nAnalysis: {analysis}\nContributions: {contributions}\nSupervisor steps: {steps}"),
            ]
        ) | model.with_structured_output(RouteDecision, method="json_schema")


async def supervisor_node(state: OrchestratorState, router: Router) -> dict:
    steps = state["supervisor_steps"] + 1
    if steps >= MAX_SUPERVISOR_STEPS:
        route = "finish"
        requested = "finish"
        reason = "step limit reached"
    else:
        decision = await router.decide(state)
        route = decision.next_agent
        requested = route
        reason = decision.reason
        if not state["reports"]:
            route = "research"
        elif state.get("analysis") is None and route == "finish":
            route = "analysis"
    return {
        "next_agent": route,
        "supervisor_steps": steps,
        "decisions": [{"step": steps, "requested": requested, "selected": route, "reason": reason}],
    }
