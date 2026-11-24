from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agents.analyst_agent import analysis_node
from app.agents.research_agent import research_node
from app.hitl import approval_node, classify_critical, rejected_node, route_after_approval, route_after_classification
from app.state import OrchestratorState
from app.supervisor import Router, supervisor_node


def initial_state(query: str) -> OrchestratorState:
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "critical": False,
        "approval_granted": None,
        "reports": [],
        "analysis": None,
        "contributions": [],
        "decisions": [],
        "next_agent": "research",
        "supervisor_steps": 0,
        "answer": "",
    }


def route_after_supervisor(state: OrchestratorState) -> Literal["research", "analysis", "final"]:
    return "final" if state["next_agent"] == "finish" else state["next_agent"]


async def final_node(state: OrchestratorState) -> dict:
    analysis = state.get("analysis")
    source = next((report for report in state["reports"] if analysis and report["source_id"] == analysis["source_id"]), None)
    if analysis and source:
        change = analysis["percent_change"]
        direction = "increase" if change > 0 else "decrease" if change < 0 else "no change"
        answer = (
            f"{analysis['source_id']}: p95 latency changed from {analysis['baseline_ms']:g} ms "
            f"to {analysis['current_ms']:g} ms ({abs(change):.1f}% {direction}). "
            f"Source: {source['path']}."
        )
    else:
        answer = "I could not verify a source-backed calculation from the available reports."
    return {"answer": answer, "next_agent": "finish"}


def build_graph(router: Router, checkpointer):
    builder = StateGraph(OrchestratorState)

    async def supervise(state: OrchestratorState) -> dict:
        return await supervisor_node(state, router)

    builder.add_node("classify", classify_critical)
    builder.add_node("approval", approval_node)
    builder.add_node("rejected", rejected_node)
    builder.add_node("supervisor", supervise)
    builder.add_node("research", research_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classification)
    builder.add_conditional_edges("approval", route_after_approval)
    builder.add_edge("rejected", END)
    builder.add_conditional_edges("supervisor", route_after_supervisor)
    builder.add_edge("research", "supervisor")
    builder.add_edge("analysis", "supervisor")
    builder.add_edge("final", END)
    return builder.compile(checkpointer=checkpointer)
