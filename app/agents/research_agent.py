import asyncio
import re
import sqlite3
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.state import OrchestratorState, ResearchReport
from app.rag.retrieval import retrieve_reports
from app.rag.store import ReportRecord


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class SearchReportsInput(BaseModel):
    query: str = Field(min_length=1)


@tool(args_schema=SearchReportsInput)
def search_reports(query: str) -> list[ResearchReport]:
    """Query a local SQLite FTS5 incident index for citable technical reports.

    Returns source IDs, relative paths, and exact source text. Use this before
    analyzing an incident; do not invent values that are absent from a report.
    """
    incident_ids = sorted(set(re.findall(r"INC-\d+", query.upper())))
    words = sorted(set(re.findall(r"[A-Za-z]{4,}", query.lower())))
    with sqlite3.connect(":memory:") as database:
        database.execute("CREATE VIRTUAL TABLE reports USING fts5(source_id UNINDEXED, path UNINDEXED, text)")
        for path in sorted(DATA_DIR.glob("*.md")):
            database.execute(
                "INSERT INTO reports(source_id, path, text) VALUES (?, ?, ?)",
                (path.stem.upper(), f"data/{path.name}", path.read_text(encoding="utf-8")),
            )
        if incident_ids:
            placeholders = ",".join("?" for _ in incident_ids)
            rows = database.execute(
                f"SELECT source_id, path, text FROM reports WHERE source_id IN ({placeholders}) ORDER BY source_id LIMIT 3",
                incident_ids,
            ).fetchall()
        elif words:
            expression = " OR ".join(f'"{word}"' for word in words)
            rows = database.execute(
                "SELECT source_id, path, text FROM reports WHERE reports MATCH ? ORDER BY bm25(reports) LIMIT 3",
                (expression,),
            ).fetchall()
        else:
            rows = []
    return [ReportRecord(source_id=source_id, path=path, text=text).model_dump() for source_id, path, text in rows]


async def research_node(state: OrchestratorState) -> dict:
    """Combine exact incident lookup with semantic retrieval for citable reports."""
    exact = await asyncio.to_thread(search_reports.invoke, {"query": state["query"]})
    dense = await retrieve_reports(state["query"])
    incident_ids = set(re.findall(r"INC-\d+", state["query"].upper()))
    if incident_ids:
        dense = [report for report in dense if report["source_id"] in incident_ids]
    reports: list[ResearchReport] = []
    seen: set[str] = set()
    for report in [*exact, *dense]:
        if report["source_id"] not in seen:
            reports.append(report)
            seen.add(report["source_id"])
    source_ids = [report["source_id"] for report in reports]
    summary = f"Found {len(reports)} matching report(s): {', '.join(source_ids) or 'none'}"
    return {
        "reports": reports,
        "analysis": None,
        "contributions": [{"agent": "research", "summary": summary, "source_ids": source_ids}],
    }
