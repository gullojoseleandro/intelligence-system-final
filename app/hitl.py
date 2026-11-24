import re
from typing import Literal

from langgraph.types import interrupt

from app.state import OrchestratorState


CRITICAL_TERMS = re.compile(
    r"\b(?:deploy|delete|erase|refund|charge|purchase|execute|production|critical|expensive)\b|high[- ]cost",
    re.IGNORECASE,
)


async def classify_critical(state: OrchestratorState) -> dict:
    """Conservatively flag requests with side effects or exceptional cost."""
    return {"critical": bool(CRITICAL_TERMS.search(state["query"]))}


def route_after_classification(state: OrchestratorState) -> Literal["approval", "supervisor"]:
    return "approval" if state["critical"] else "supervisor"


async def approval_node(state: OrchestratorState) -> dict:
    """Pause a critical request until a human submits an explicit decision."""
    decision = interrupt({"reason": "critical operation", "query": state["query"]})
    return {"approval_granted": decision.get("approved") is True}


def route_after_approval(state: OrchestratorState) -> Literal["supervisor", "rejected"]:
    return "supervisor" if state["approval_granted"] else "rejected"


async def rejected_node(state: OrchestratorState) -> dict:
    return {"answer": "Human approval declined; no specialist work was performed."}
